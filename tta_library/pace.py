"""
Copyright to PACE Authors ECML PKDD 2026
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from typing import List
from models.vpt import PromptViT
import cma
import numpy as np
import os
from dataclasses import dataclass, field
from utils.fastfood import FastfoodOrthogonal

from utils.utils import get_transformer_hidden_size
from quant_library.quant_layers.matmul import *
from tta_library.tta_method import TTAMethod
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy



class DomShiftDet:
    def __init__(self, domain_t=0.1, bs=64) -> None:
        self.reset_domain_info()
        self.update_domain_info = True
        self.domain_t = domain_t
        self.bs = bs
    
    def reset_domain_info(self):
        self.domain_var = None
        self.domain_mean = None

    def domain_shift(self, embed_features):
        # do not detect shift
        if self.domain_t == -1.0:
            return 0.0, False
        
        dom_shift_val, domain_change = self._check_should_save(embed_features)
        if domain_change:
            self.reset_domain_info()

        if self.update_domain_info:
            self._update_domain_info(embed_features)
        return dom_shift_val, domain_change
    
    @torch.no_grad()
    def _check_should_save(self, embed_features):
        if embed_features.shape[0] < self.bs: # small batch size leads to disturbulance
            return float('nan'), False
        
        if self.domain_var is None or self.domain_mean is None:
            return float('nan'), False
        
        emb_var, emb_mean = embed_features.var(dim=(0,1)), embed_features.mean(dim=(0,1))

        dom_shift = self._calculate_domain_shift(
            self.domain_var,
            self.domain_mean,
            emb_var,
            emb_mean
        )
        
        return dom_shift.item(), dom_shift > self.domain_t # there is a shift, default 0.1 or 0.05
    
    @torch.no_grad()
    def _calculate_domain_shift(self, domain_var, domain_mean, cur_var, cur_mean, eps=1e-8):
        d1 = (domain_var + (domain_mean - cur_mean) ** 2) / 2. / (cur_var + eps) - 0.5
        d2 = (cur_var + (domain_mean - cur_mean) ** 2) / 2. / (domain_var + eps) - 0.5
        return torch.mean((d1+d2))

    @torch.no_grad()
    def _update_domain_info(self, embed_features):
        if embed_features.shape[0] < self.bs: # small batch size leads to disturbulance
            return False
        
        emb_var, emb_mean = embed_features.var(dim=(0,1)), embed_features.mean(dim=(0,1))
        if self.domain_var is None:
            self.domain_var, self.domain_mean = emb_var, emb_mean
        else:
            self.domain_var = 0.8 * self.domain_var + 0.2 * emb_var
            self.domain_mean = 0.8 * self.domain_mean + 0.2 * emb_mean


@dataclass
class VectorBank:
    max_len: int
    means: List[np.ndarray] = field(default_factory=list)
    hist_stats: List[torch.Tensor] = field(default_factory=list)
    
    def _remove_entry(self, idx):
        self.means.pop(idx)
        self.hist_stats.pop(idx)
    
    def _add_entry(self, mean: np.ndarray, hist_stat: torch.Tensor):
        if self.max_len < 1:
            return
        
        # NOTE: verify the dimensionality of mean
        self.means.append(deepcopy(mean))
        self.hist_stats.append(hist_stat)
        
        if len(self) > self.max_len:
            # Prune the most redundant entry based on cosine similarity between CMA means.
            M = self.get_means() # [N, D]
            eps = 1e-12
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            M_unit = M / np.maximum(norms, eps)

            # Cosine similarity matrix S = M_unit @ M_unit^T
            S = M_unit @ M_unit.T  # [N, N]
            np.fill_diagonal(S, 0.0)  # ignore self-similarity

            if S.shape[0] > 1:
                avg_sim = S.sum(axis=1) / (S.shape[0] - 1)
                remove_idx = int(np.argmax(avg_sim))
            else:
                remove_idx = 0

            self._remove_entry(remove_idx)
    
    def get_means(self) -> np.ndarray:
        return np.stack(self.means, axis=0)  # [N, D]

    def get_hist_stats(self) -> List[torch.Tensor]:
        return self.hist_stats

    def __len__(self) -> int:
        assert len(self.means) == len(self.hist_stats) 
        return len(self.means)

class PACE(TTAMethod):
    def __init__(self, model:PromptViT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()
        self.transformer_hidden_size = get_transformer_hidden_size(self.model)

        self.updated_params, self.updated_param_names = self.get_updated_params()
        self.source_model_dict = self.model.state_dict()
        
        self.updated_dim  = sum([param.data.numel() for param in self.updated_params])

        print(f"\nUpdated dim: {self.updated_dim}")

        self.proj = FastfoodOrthogonal(self.args.cma_optim_dim, self.updated_dim)

        self.base_vec = parameters_to_vector(self.updated_params).detach().clone()
        self.work_vec = self.base_vec.clone()  # reusable buffer to avoid allocations

        cma = self._init_cma(
            mean=self.args.cma_optim_dim * [0],
            sigma=self.args.cma_init_sigma
        )
        self.set_cma(cma)

        self.best_prompts = np.zeros((self.args.cma_optim_dim,))
        self.best_loss = np.inf
        self.hist_stat = None
        
        self.stop_adapt = False

        self.do_prev_domain_cma_init = False
        self.history_buffer = VectorBank(max_len=self.args.vector_bank_size)

        self.dom_shift_det = DomShiftDet(domain_t=self.args.gamma,
                                                bs=self.args.batch_size)
        
    def _init_cma(self, mean, sigma):
        """CMA-ES initialization"""
        cma_opts = {
            'seed': 2020,
            'popsize': self.args.cma_pop_size,
            'maxiter': -1,
            'verbose': -1,
        }

        es = cma.CMAEvolutionStrategy(self.args.cma_optim_dim * [0], 
                                      self.args.cma_init_sigma,
                                      inopts=cma_opts)
        return es
    
    def set_cma(self, es):
        self.es = es

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-self.transformer_hidden_size:] - self.hist_stat
        
    def handle_domain_shift(self, x, y=None):
        # 1. SELECTION: Find best historical matches
        # `HistoryBuffer.get_means()` returns shape [N, D] (N history entries, D CMA dim).
        # If the buffer is empty, start with an empty [0, D] so we can still append the
        # "initial prompt" (all zeros) below.
        if len(self.history_buffer) == 0:
            means = np.empty((0, int(self.args.cma_optim_dim)), dtype=np.float64)
        else:
            means = self.history_buffer.get_means()
        
        # add intial prompt
        means = np.asarray(means)
        means_2d = np.atleast_2d(means)  # [N, D] (also handles accidental 1D input)
        zero_row = np.zeros((1, means_2d.shape[1]), dtype=means_2d.dtype)  # [1, D]
        means = np.concatenate([means_2d, zero_row], axis=0)  # [N+1, D]

        deltas = self._project_prompts(means)
        
        scores = []
        for delta, mean in zip(deltas, means):
            shift_vector = None
            
            self._apply_delta(delta)
    
            _, loss, _, _ = forward_and_get_loss(x, self.model, self.args.fitness_lambda, self.train_info, shift_vector,
                                                            self.imagenet_mask, self.transformer_hidden_size, gt=y)

            self._reset_model()
    
            scores.append((loss, mean))
        
        # Sort by lowest loss
        scores.sort(key=lambda x: x[0])
        new_mean = scores[0][1]
        
        top_means = []
        if len(scores) > 1:
            top_means = [s[1] for s in scores[:4]]
        
        # 3. INITIALIZATION: Create the optimizer
        cma = self._init_cma(mean=new_mean, sigma=self.args.cma_init_sigma)
        self.set_cma(cma)

        # 4. INJECTION: Override the first population
        # Ask for standard solutions
        population = self.es.ask()
        
        # Inject historical means into the first few slots
        for i, past_mean in enumerate(top_means):
            population[i] = past_mean
        
        return population

    def forward(self, x, y=None):
        """calculating shift direction, Eqn. (8)"""
        shift_vector = self._get_shift_vector()

        if not self.stop_adapt:
    
            """Sampling from CMA-ES and evaluate the new solutions.
            Note that we also compare the current solutions with the previous best one"""
            if self.do_prev_domain_cma_init and self.args.vector_bank_size > 0:
                prompts = self.handle_domain_shift(x)
                self.do_prev_domain_cma_init = False
            else:
                prompts = self.es.ask() + [self.best_prompts]

            self.best_loss, self.best_outputs, batch_means = np.inf, None, []

            losses = []

            with torch.inference_mode():
                deltas = self._project_prompts(prompts)  # [pop, updated_dim]

                for i, delta in enumerate(deltas):
                    self._apply_delta(delta)
            
                    outputs, loss, batch_mean, stem_embed = forward_and_get_loss(x, self.model, self.args.fitness_lambda, self.train_info, shift_vector,
                                                                    self.imagenet_mask, self.transformer_hidden_size, gt=y)
                    batch_means.append(batch_mean[-self.transformer_hidden_size:].unsqueeze(0))
                    del batch_mean
                    
                    if self.best_loss > loss.item():
                        self.best_prompts = deepcopy(prompts[i])
                        self.best_loss = loss.item()
                        self.best_outputs = outputs.clone()
                        outputs = None
                    losses.append(loss.item())
                    del outputs

                    self._reset_model()

        else:
            outputs, loss, batch_mean, stem_embed = forward_and_get_loss(x, self.model, self.args.fitness_lambda, self.train_info, shift_vector,
                                                            self.imagenet_mask, self.transformer_hidden_size, gt=y)
            self.best_outputs = outputs
            self.best_loss = loss
            batch_means = [batch_mean[-self.transformer_hidden_size:].unsqueeze(0)]
            

        """CMA-ES updates, Eqn. (6)"""
        prev_mean: np.ndarray = np.asarray(self.es.mean).copy()
        if not self.stop_adapt:
            self.es.tell(prompts, losses)
        curr_mean = np.asarray(self.es.mean)
        mean_diff = np.linalg.norm(curr_mean - prev_mean)
        denom = (np.linalg.norm(prev_mean) * np.linalg.norm(curr_mean)) + 1e-12
        cos_sim = float(np.dot(prev_mean, curr_mean) / denom)
        cma_mean_diff_normed =  float(mean_diff / (np.linalg.norm(prev_mean) + 1e-13))
        
        """Update overall test statistics, Eqn. (9)"""
        batch_means = torch.cat(batch_means, dim=0).mean(0)
            
        self._update_hist(batch_means)

        stem_kl_div, is_shift = self.dom_shift_det.domain_shift(stem_embed)
        
        if is_shift:
            self.reset()
        elif self.stop_adapt_criteria(cos_sim, cma_mean_diff_normed):
            self.handle_stop_adapt()
        

        self.tb_logger.log_scalars('per_batch', {
            'best_loss': self.best_loss,
            'cma_sigma': self.es.sigma,
            'cma_mean_prev_diff_norm_normed': cma_mean_diff_normed,
            'cma_mean_prev_cos_sim': cos_sim,
            'domain_shift_detected': int(is_shift),
            'zoa_stem_kl_div': stem_kl_div,
            'stop_adapt': int(self.stop_adapt),
        })
        
        return self.best_outputs
    
    def stop_adapt_criteria(self, cos_sim, cma_mean_diff_normed):
        return not self.stop_adapt and self.args.epsilon > 0 and cma_mean_diff_normed < self.args.epsilon
        
    def handle_stop_adapt(self):
        self.stop_adapt = True
        self.dom_shift_det.update_domain_info = False

        delta = self._project_prompts([self.es.mean])[0]
        self._apply_delta(delta) 
    
    def obtain_origin_stat(self, train_loader):
        self.model.eval()
        print('===> begin calculating mean and variance')
        features = []
        with torch.no_grad():
            for _, dl in enumerate(train_loader):
                images = dl[0].cuda()
                feature = self.model.layers_cls_features(images)
                features.append(feature)
            features = torch.cat(features, dim=0)
            self.train_info = torch.std_mean(features, dim=0) # occupy 0.2MB 
        del features

        # preparing quantized model for prompt adaptation
        for _, m in self.model.vit.named_modules():
            if type(m) == PTQSLBatchingQuantMatMul:
                m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_prompts,64)).cuda(), torch.zeros((1,12,64,197+self.model.num_prompts)).cuda())
            elif type(m) == SoSPTQSLBatchingQuantMatMul:
                m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_prompts,197+self.model.num_prompts)).cuda(), torch.zeros((1,12,197+self.model.num_prompts,64)).cuda())
        print('===> calculating mean and variance end')

    def reset(self):
        self.history_buffer._add_entry(getattr(self.es, "mean"), self.hist_stat)
        cma = self._init_cma(
            mean=self.args.cma_optim_dim * [0],
            sigma=self.args.cma_init_sigma
        )
        self.set_cma(cma)
        self.hist_stat = None

        self.best_prompts = np.zeros((self.args.cma_optim_dim,))
        self.stop_adapt = False
        self.dom_shift_det.update_domain_info = True
        self.do_prev_domain_cma_init = True

        self._reset_model()

    @staticmethod
    def is_fixed_layer(name):
        return any(name == prefix or name.startswith(prefix + ".") or name.startswith(prefix.replace('model.', '')) \
            for prefix in ['model.vit.blocks.0.norm1', 'model.vit.blocks.0.norm2', 'model.vit.blocks.9', 'model.vit.blocks.10', 
                           'model.vit.blocks.11', 'model.vit.norm'])

    def get_updated_params(self):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        """
        params = []
        names = []
        for nm, m in self.named_modules():
            if self.is_fixed_layer(nm):
                continue
                
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def _reset_model(self) -> None:
        # restore once at the end (if you need the model back to source)
        vector_to_parameters(self.base_vec, self.updated_params)

    def _project_prompts(self, prompts: List[np.ndarray]) -> torch.Tensor:
        prompts_t = torch.as_tensor(np.asarray(prompts), device="cuda", dtype=torch.float32)  # [pop, optim_dim]
        return self.proj(prompts_t)

    def _apply_delta(self, delta: torch.Tensor) -> None:
        assert delta.shape == self.base_vec.shape
        self.work_vec.copy_(self.base_vec).add_(delta)     # no alloc
        vector_to_parameters(self.work_vec, self.updated_params)  # writes directly into model params


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

@torch.jit.script
def softmax_entropy_gt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(y * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='none').cuda()

def forward_and_get_loss(images, model:PromptViT, fitness_lambda, train_info, shift_vector, imagenet_mask, transformer_hidden_size,
                         gt=None):
    features, stem_embed = model.layers_cls_features_with_prompts_stem_layer(images)

    """discrepancy loss for Eqn. (5)"""
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    # NOTE: $lambda$ should be 0.2 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64
    
    cls_features = features[:, -transformer_hidden_size:] # the feature of classification token
    output = model.vit.head(cls_features)

    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    
    entropy_loss = softmax_entropy(output).sum()
    loss = discrepancy_loss + entropy_loss
    
    """activation shifting, Eqn. (7)"""
    if shift_vector is not None:
        output = model.vit.head(cls_features + 1. * shift_vector)
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean, stem_embed
