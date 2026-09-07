import argparse
import math
import os
import random
import time
from importlib import import_module, reload

import numpy as np
import timm
import torch
import torch.nn.functional as F

from dataset.domainnet import DomainNet126
import tta_library.cotta as cotta
import tta_library.pace as pace
import tta_library.sar as sar
import tta_library.tent as tent
import tta_library.zoa_vit as zoa_vit
from calibration_library.metrics import ECELoss
from dataset.ImageNetMask import imagenet_r_mask
from dataset.selectedRotateImageFolder import prepare_test_data, prepare_train_dataloader, prepare_train_dataset
from models.vpt import PromptViT
from tta_library.foa import FOA
from tta_library.lame import LAME
from tta_library.sam import SAM
from tta_library.t3a import T3A
from tta_library.source import Source
from utils.cli_utils import *
from utils.tensorboard_logger import TensorBoardLogger
from utils.utils import get_logger, VRAMTracker
from models.fuse_vit import FuseViT
from quant_model import create_quant_model
from utils.utils import get_num_classes


def validate_adapt(val_loader, model, args, tb_logger: TensorBoardLogger, vram_tracker: VRAMTracker):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5, vram_tracker],
        prefix='Test: ')
    
    outputs_list, targets_list = [], []
    
    with torch.no_grad():
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]

            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            
            start = time.time()
            output = model(images, target)
            batch_time.update(time.time() - start)

            # for calculating Expected Calibration Error (ECE)
            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            del output
            
            tb_logger.log_scalars(
                'per_batch',
                {
                    'acc1': acc1.item(),
                },
            )
            tb_logger.step += 1

            # measure elapsed time
            vram_tracker.update()
            if i % 5 == 0:
                logger.info(progress.display(i))
            
        outputs_list = torch.cat(outputs_list, dim=0).numpy()
        targets_list = torch.cat(targets_list, dim=0).numpy()
        
        logits = args.algorithm != 'lame' # only lame outputs probability
        
        tb_logger.step -= 1

        ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits) # calculate ECE

        tb_logger.log_scalars('per_domain', {'ece': ece_avg})
        tb_logger.log_scalars('per_domain', {'acc': top1.avg})
        tb_logger.log_scalars('per_domain', {'time_avg_per_batch': batch_time.avg})
        tb_logger.log_scalars('per_domain', {'time_sum': batch_time.sum})
        tb_logger.step += 1
        
        logger.info(
            f"Domain acc: {top1.avg}"
        )
        logger.info(
            f"Domain ece: {ece_avg}"
        )
    
    return top1.avg, top5.avg, ece_avg, batch_time.sum

def obtain_train_loader(args):
    args.corruption = 'original'
    train_dataset, train_loader = prepare_test_data(args)
    if hasattr(train_dataset, 'switch_mode'):
        train_dataset.switch_mode(True, False)
    return train_dataset, train_loader

def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./quant_lib/configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"quant_lib.configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--data_rendition', default='/dockerdata/imagenet-r', help='path to corruption dataset')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')
    
    parser.add_argument('--arch', default='vit_base', choices=['vit_base', 'deit'], type=str, help='model architecture')
    parser.add_argument('--resume', default=False, action='store_true', help='whether to load the quantized model')

    # algorithm selection
    parser.add_argument('--algorithm', default='foa', type=str, help='supporting foa, sar, cotta and etc.')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    parser.add_argument(
        '--num_loops',
        default=1,
        type=int,
        help='number of times to iterate through the corruption list sequentially (useful for repeated runs)',
    )
    parser.add_argument('--dataset', default='imagenet_c', type=str, help='')
    parser.add_argument('--dataroot', default='/datasets', type=str, help='')

    # model settings
    parser.add_argument('--quant', default=False, action='store_true', help='whether to use quantized model in the experiment')
    parser.add_argument('--bit', default=8, type=int, help='the bit width of the quantized model')

    # foa settings
    parser.add_argument('--num_prompts', default=3, type=int, help='number of inserted prompts for test-time adaptation.')    
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='the balance factor $lambda$ in FOA')    
    parser.add_argument('--lambda_bp', default=30, type=float, help='the balance factor $lambda$ in FOA-BP')    
    
    parser.add_argument('--univ', default=None, type=float)    

    # compared method settings
    parser.add_argument('--margin_e0', default=0.4*math.log(1000), type=float, help='the entropy margin for sar')    

    # output settings
    parser.add_argument('--output', default='./outputs', help='the output directory of this experiment')
    parser.add_argument('--tag', default='_first_experiment', type=str, help='the tag of experiment')

    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')

    # ZOA settings
    parser.add_argument('--lr_alpha', '--lra', default=0.01, type=float, help='learning rate for alpha (default: 0.1)')
    parser.add_argument('--spsa_c', '--sc',default=0.01, type=float, help='the c factor of spsa, step size of perturbation')
    parser.add_argument('--spsa_c_alpha', '--sca',default=0.05, type=float, help='the c factor of spsa for alpha')
    parser.add_argument('--spsa_momentum', '--sm', default=0, type=float, help='the momentum factor of spsa_gc')    
    parser.add_argument('--weight_decay', '--wd', default=0.4, type=float, help='the weight decay factor of optimizer')   
    parser.add_argument('--weight_decay_alpha', '--wda', default=0.1, type=float, help='the weight decay factor of optimizer for alpha')   
    parser.add_argument('--sp_avg', '--avg', default=1, type=int, help='the number of rounds for spsa, 27 for ZOA (FP=28)') 
    parser.add_argument('--max_weight_nums', '--mwn', default=32, type=int, help='the maximun number of the domain weights')
    parser.add_argument('--alpha_scale', default=10., type=float, help='"Scale the gradients based on the desired max norm')
    parser.add_argument('--domain_t', '--dt', default=-1.0, type=float, help='the threshold for the minimum specificity')
    
    # PACE settings
    parser.add_argument('--cma_init_sigma', default=0.05, type=float, help='initial step-size (sigma) for CMA-ES')
    parser.add_argument('--cma_pop_size', default=27, type=int, help='population size for CMA-ES')
    parser.add_argument('--cma_optim_dim', default=2304, type=int, help='dimensionality of the CMA-ES search space (projected onto the norm-layer params via Fastfood)')
    parser.add_argument('--vector_bank_size', default=0, type=int, help='max number of past CMA-ES distribution means kept in the vector bank, p in the paper; 0 disables reuse on domain shift')
    parser.add_argument('--gamma', default=-1.0, type=float, help='threshold on the domain-shift statistic above which a domain shift is detected, \gamma in the paper; -1 disables detection')
    parser.add_argument('--epsilon', default=-1.0, type=float, help='threshold on the normalized CMA-ES mean drift below which adaptation is stopped, \epsilon in the paper (Eq. 6); <=0 disables early stopping')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # create logger for experiment
    args.output += '/' + args.algorithm + args.tag + '/'
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    logger = get_logger(name="project", 
                        output_directory=args.output, 
                        log_name="log_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".txt", 
                        debug=False)
    logger.info(args)

    tb_logger = TensorBoardLogger(args.output)

    # configure the domains for adaptation
    if args.dataset == "imagenet_c":
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    elif args.dataset == "imagenet_r":
        corruptions = ['rendition']
    elif args.dataset == "domainnet126":
        corruptions = ["clipart", "painting", "sketch"]
        # corruptions = ["real"]
    else:
        raise NotImplementedError(args.dataset)
        
    if args.arch == 'vit_base':
        if args.dataset == 'domainnet126':
            net = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
            net.head = torch.nn.Linear(net.embed_dim, get_num_classes(args.dataset))

            # source_domain = 'real'
            source_domain = [dom for dom in DomainNet126.ENVIRONMENTS if dom not in corruptions][0]
            assert isinstance(source_domain, str)
            state_dict = torch.load(os.path.join('ckpts/domainnet126/ViT-B16', source_domain, 'model.pkl'))['model_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'featurizer' in k or 'classifier' in k:
                    continue

                key_mod = k[len('network.module.0.'):] if k.startswith('network.module.0.') else k
                key_mod = key_mod.replace('module.1','head') if key_mod.startswith('network.module.1.') else key_mod
                key_mod = key_mod[len('network.'):] if key_mod.startswith('network.') else key_mod
                if key_mod in new_state_dict:
                    raise KeyError(f"Duplicate key after stripping 'featurizer.': {key_mod}")
                new_state_dict[key_mod] = v
                
            state_dict = new_state_dict

            net.load_state_dict(state_dict)
            net.cuda()
        else:
            net = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
    elif args.arch == 'deit':
        if args.dataset == 'domainnet126':
            net = timm.create_model('deit3_base_patch16_224', pretrained=True)
            net.head = torch.nn.Linear(net.embed_dim, get_num_classes(args.dataset))
            if hasattr(net, 'head_dist'):
                # Some DeiT variants have a distillation head which must remain
                # callable inside timm's forward(). Use Identity instead of None.
                net.head_dist = torch.nn.Identity()
            
            source_domain = [dom for dom in DomainNet126.ENVIRONMENTS if dom not in corruptions][0]
            assert isinstance(source_domain, str)
            state_dict = torch.load(os.path.join('ckpts/domainnet126/DeiT', source_domain, 'model.pkl'))['model_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'featurizer' in k or 'classifier' in k:
                    continue

                key_mod = k[len('network.module.0.'):] if k.startswith('network.module.0.') else k
                key_mod = key_mod.replace('module.1','head') if key_mod.startswith('network.module.1.') else key_mod
                key_mod = key_mod[len('network.'):] if key_mod.startswith('network.') else key_mod
                if key_mod in new_state_dict:
                    raise KeyError(f"Duplicate key after stripping 'featurizer.': {key_mod}")
                new_state_dict[key_mod] = v
                
            state_dict = new_state_dict
            net.load_state_dict(state_dict)
            net.cuda()
        else:
            net = timm.create_model('deit3_base_patch16_224', pretrained=True).cuda()
    else:
        raise NotImplementedError

    if args.quant:
        net = create_quant_model(args, logger, net, None)
        
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)

    if args.algorithm == 'tent':
        net = tent.configure_model(net)
        params, _ = tent.collect_params(net)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) # lr=0.001
        adapt_model = tent.Tent(net, optimizer, 
                                logger=logger, 
                                tb_logger=tb_logger,
                                univ=args.univ)
    elif args.algorithm == 'foa':
        net = PromptViT(net, args.num_prompts).cuda()
        adapt_model = FOA(net,
                        logger=logger, 
                        tb_logger=tb_logger,
                        args=args,
                        univ=args.univ)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'pace':
        # no prompts needed
        net = PromptViT(net, 0).cuda()
        adapt_model = pace.PACE(net,
                                logger=logger, 
                                tb_logger=tb_logger,
                                args=args,
                                univ=args.univ)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 't3a':
        adapt_model = T3A(net, get_num_classes(args.dataset), 20,
                                logger=logger, 
                                tb_logger=tb_logger,
                                univ=args.univ).cuda()
    elif args.algorithm == 'sar':
        net = sar.configure_model(net)
        params, _ = sar.collect_params(net)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
        # NOTE: set margin_e0 to 0.4*math.log(200) on ImageNet-R
        adapt_model = sar.SAR(net, optimizer, margin_e0=args.margin_e0,
                                logger=logger, 
                                tb_logger=tb_logger,
                                univ=args.univ)
    elif args.algorithm == 'cotta':
        net = cotta.configure_model(net)
        params, _ = cotta.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
        adapt_model = cotta.CoTTA(net, optimizer, steps=1, episodic=False,
                                logger=logger, 
                                tb_logger=tb_logger,
                                univ=args.univ)
    elif args.algorithm == 'lame':
        adapt_model = LAME(net,
                                logger=logger, 
                                tb_logger=tb_logger,
                                univ=args.univ)
    elif args.algorithm == 'zoa_vit':
        net = FuseViT(args, net, logger=logger)
        net = zoa_vit.configure_model(net).cuda()
        net.replace_fuse_ln()
        net.configure_model()

        params = net.collect_params()
        alpha_optimizer = torch.optim.AdamW([
            {'params': params['alpha'], 'lr': args.lr_alpha, 'weight_decay': args.weight_decay_alpha}
        ])

        eps_optimizer = torch.optim.SGD(params['epsilon_weight'] + params['epsilon_bias'], 
                                        args.lr, momentum=args.spsa_momentum, weight_decay=args.weight_decay)
        net.save_root = args.output
        net.set_optimizers(alpha_optimizer, eps_optimizer)

        adapt_model = zoa_vit.ZOA_ViT(net, 
                                      logger=logger, 
                                    tb_logger=tb_logger,
                                    args=args,
                                    univ=args.univ)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'no_adapt':
        adapt_model = Source(net,
                                logger=logger, 
                                tb_logger=tb_logger,
                                univ=args.univ)
    else:
        assert False, NotImplementedError


    # Initialize VRAM tracker
    vram_tracker = VRAMTracker()
    torch.cuda.reset_peak_memory_stats()
    
    corrupt_acc_all, corrupt_top5_all, corrupt_ece_all = [], [], []
    num_loops = max(1, int(getattr(args, 'num_loops', 1)))
    time_overall = 0

    for rep_idx in range(num_loops):
        time_loop = 0
        
        corrupt_acc_rep, corrupt_top5_rep, corrupt_ece_rep = [], [], []
        logger.info(f"Starting corruption sequence repetition {rep_idx + 1}/{num_loops}")

        for corrupt in corruptions:
            args.corruption = corrupt
            logger.info(f"[rep {rep_idx + 1}/{num_loops}] {args.corruption}")

            if args.corruption == 'rendition':
                adapt_model.imagenet_mask = imagenet_r_mask
            else:
                adapt_model.imagenet_mask = None

            val_dataset, val_loader = prepare_test_data(args)

            # dataset = prepare_train_dataset(args)
            # val_loader, _ = prepare_train_dataloader(args, trset=dataset)

            torch.cuda.empty_cache()
            top1, top5, ece_loss, time_sum = validate_adapt(val_loader, adapt_model, args, tb_logger, vram_tracker)
            logger.info(
                f"Under shift type {args.corruption} (rep {rep_idx + 1}/{num_loops}) After {args.algorithm} "
                f"Top-1 Accuracy: {top1:.6f} and Top-5 Accuracy: {top5:.6f} and ECE: {ece_loss:.6f}"
            )
            
            time_loop += time_sum
            time_overall += time_sum

            corrupt_acc_all.append(top1)
            corrupt_ece_all.append(ece_loss)
            corrupt_acc_rep.append(top1)
            corrupt_ece_rep.append(ece_loss)

            logger.info(
                f"[rep {rep_idx + 1}/{num_loops}] mean acc so far: "
                f"{sum(corrupt_acc_rep)/len(corrupt_acc_rep) if len(corrupt_acc_rep) else 0}"
            )
            logger.info(
                f"[rep {rep_idx + 1}/{num_loops}] mean ece so far: "
                f"{sum(corrupt_ece_rep)/len(corrupt_ece_rep)*100 if len(corrupt_ece_rep) else 0}"
            )

        # TensorBoard: per-loop summary
        tb_logger.log_scalars(
            "per_loop",
            {
                "acc1": sum(corrupt_acc_rep) / len(corrupt_acc_rep) if len(corrupt_acc_rep) else 0,
                "ece": sum(corrupt_ece_rep) / len(corrupt_ece_rep) * 100 if len(corrupt_ece_rep) else 0,
                "time": time_loop,
            },
        )

    overall_acc1 = sum(corrupt_acc_all) / len(corrupt_acc_all) if len(corrupt_acc_all) else 0
    overall_ece = sum(corrupt_ece_all) / len(corrupt_ece_all) * 100 if len(corrupt_ece_all) else 0

    logger.info(
        f"Overall acc: {overall_acc1}"
    )
    logger.info(
        f"Overall ece: {overall_ece}"
    )
    logger.info(
        f"Overall time: {time_overall}"
    )

    tb_logger.log_scalars(
        "overall",
        {
            "acc1": overall_acc1,
            "ece": overall_ece,
            "time": time_overall,
        },
    )        
    
    # Log peak VRAM usage at the end of experiment
    vram_tracker.log_peak_usage(logger, "Experiment completed - ")
    
    # Log peak VRAM to TensorBoard
    max_allocated, max_allocated_cuda, max_cached = vram_tracker.get_peak_usage()
    tb_logger.log_scalars("overall", {
        "peak_vram_allocated_mb": max_allocated,
        "peak_vram_allocated_cuda_mb": max_allocated_cuda,
        "peak_vram_cached_mb": max_cached,
    })
