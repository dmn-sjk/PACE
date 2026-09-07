import os
import sys
import logging
import random

import numpy as np
import torch

import torch.nn as nn
from typing import Callable

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def mean(items):
    return sum(items)/len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def get_transformer_hidden_size(model):
    if hasattr(model, "vit") and hasattr(model.vit, "embed_dim"):
        return model.vit.embed_dim
    if hasattr(model, "prompt_dim"):
        return model.prompt_dim
    if hasattr(model, "embed_dim"):
        return model.embed_dim
    raise AttributeError("Unable to infer transformer hidden size from model architecture")


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger
    

def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_flops(module: nn.Module, size, skip_pattern, device):
    # print(module._auxiliary)
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("init hool for", name)
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        # print(f"training={training}")
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops

def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param


def get_vram_usage():
    """
    Get current VRAM usage in MB.
    Returns a tuple of (allocated_mb, cached_mb, total_mb)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        cached = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # Convert to MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        return allocated, cached, total, max_allocated
    else:
        return 0, 0, 0, 0


class VRAMTracker:
    """
    Tracks peak VRAM usage during experiment.
    """
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.max_allocated = 0
        self.max_cached = 0
        self.max_allocated_cuda = 0
    
    def update(self):
        """Update peak VRAM usage."""
        if torch.cuda.is_available():
            allocated, cached, _, max_allocated_cuda = get_vram_usage()
            self.max_allocated = max(self.max_allocated, allocated)
            self.max_allocated_cuda = max(self.max_allocated_cuda, max_allocated_cuda)
            self.max_cached = max(self.max_cached, cached)
    
    def get_peak_usage(self):
        """Get peak VRAM usage."""
        return self.max_allocated, self.max_allocated_cuda, self.max_cached
    
    def log_peak_usage(self, logger, prefix=""):
        """Log peak VRAM usage."""
        if torch.cuda.is_available():
            logger.info(f"{prefix}Peak VRAM - \
                Allocated: {self.max_allocated:.1f}MB, \
                Max Allocated: {self.max_allocated_cuda:.1f}MB, \
                Cached: {self.max_cached:.1f}MB")
        else:
            logger.info(f"{prefix}VRAM - CUDA not available")
    
    def __str__(self):
        fmtstr = 'Peak VRAM - \
            Allocated: {max_allocated:.1f}MB, \
            Max Allocated: {max_allocated_cuda:.1f}MB, \
            Cached: {max_cached:.1f}MB'
        return fmtstr.format(max_allocated=self.max_allocated,
                             max_allocated_cuda=self.max_allocated_cuda, 
                             max_cached=self.max_cached)

def get_projection_from_gradients(G, optim_dim, return_evr: bool = False):
    N,D = G.shape
    print(f"Input dimensions: N={N}, D={D}")
    
    # 1. DO NOT CENTER G
    # We preserve the raw gradient vectors so the mean update direction 
    # is captured as the primary component.

    # 2. Compute the Gram Matrix (The 'Snapshot')
    # This measures how similar every gradient is to every other gradient.
    # Shape: [784, 784]
    print("Computing Gram Matrix...")
    K = torch.matmul(G, G.T)

    # 3. Eigendecomposition
    # We use 'eigh' because K is symmetric.
    # S: Eigenvalues (ascending), V: Eigenvectors
    print("Computing Eigenvectors...")
    S, V = torch.linalg.eigh(K)

    # 4. Sort Descending (Highest variance first)
    # We want the most important directions at index 0
    idx = torch.argsort(S, descending=True)
    S = S[idx]
    V = V[:, idx]

    # Explained variance ratio (EVR) from eigenvalues.
    # Note: K = G G^T is PSD in theory, but can have tiny negative eigenvalues due to numerics.
    S_pos = torch.clamp(S, min=0.0)
    evr = S_pos / (S_pos.sum() + 1e-12)

    # 4. Enforce k limit
    # We take the minimum of k or the actual number of samples N
    assert optim_dim <= N
    actual_k = optim_dim
    # actual_k = min(optim_dim, N)
    
    # Slice the arrays to keep only the top k
    S = S[:actual_k]      # Shape [k]
    V = V[:, :actual_k]   # Shape [N, k]
    
    print(f"Selecting top {actual_k} components.")

    # 5. Filter Numerical Noise (Optional but recommended)
    # Keep components where eigenvalue > tiny threshold
    valid_mask = S > 1e-5
    S = S[valid_mask]
    V = V[:, valid_mask]
    
    num_components = S.shape[0]
    print(f"kept {num_components} components (100% variance).")

    # 6. Recover the High-Dimensional Basis (P)
    # Formula: P = G.T @ V @ S^(-1/2)
    # This maps the low-dim eigenvectors back to 38400-dim space
    # and normalizes them to be unit length.
    
    sigma_inv = 1.0 / torch.sqrt(S)
    
    # We broadcast multiply V by the inverse singular values
    V_scaled = V * sigma_inv.unsqueeze(0) 
    
    # Project back
    # [38400, 784] = [38400, 784] @ [784, k]
    print("Projecting to Parameter Space...")
    P = torch.matmul(G.T, V_scaled)
    W_up = P.T
    if return_evr:
        return W_up, evr.detach().cpu().numpy()
    return W_up

def get_projection_from_saved_gradients(bp_grads_path, optim_dim, is_fixed_layer_func: Callable[[str], bool]):
    if isinstance(bp_grads_path, (str, os.PathLike)):
        bp_grads_paths = [bp_grads_path]
    else:
        bp_grads_paths = bp_grads_path

    gradients = []
    for path in bp_grads_paths:
        bp_grads = torch.load(path, map_location='cpu')
        G = torch.cat([t for name, t in bp_grads.items() if not is_fixed_layer_func(name)], dim=1).cuda()
        gradients.append(G)

    G = torch.cat(gradients, dim=0).cuda()

    return get_projection_from_gradients(G, optim_dim)

    
def get_num_classes(dataset_name: str):
    dataset_name2num_classes = {"imagenet_c": 1000, "imagenet_r": 200, "domainnet126": 126
                                }
    assert dataset_name in dataset_name2num_classes.keys(), \
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(dataset_name2num_classes.keys())}"
    return dataset_name2num_classes[dataset_name]