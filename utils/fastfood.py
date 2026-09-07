import torch
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform


class FastfoodOrthogonal(nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # We need N=65536 total, but we will process it in blocks of 32768
        self.N = 2**((max(input_dim, output_dim) - 1).bit_length())
        self.block_size = 32768 
        self.num_blocks = self.N // self.block_size # This will be 2 for your case
        
        # Random components
        self.register_buffer('B', torch.randint(0, 2, (self.N,), device=device).float() * 2 - 1)
        self.register_buffer('P', torch.randperm(self.N, device=device))
        self.register_buffer('G', torch.randint(0, 2, (self.N,), device=device).float() * 2 - 1)
        
        # Scaling to maintain orthonormality (1/N total across two H steps)
        # self.scale = 1.0 / self.N

        # 1. Initialize scale to 1.0 temporarily
        self.scale = 1.0
        # 2. Run a probe vector to measure energy loss
        with torch.no_grad():
            # Create a random vector with norm 1
            probe = torch.randn(self.input_dim, device=device)
            probe = probe / probe.norm()
            
            # Run the full forward pass
            # (Make sure to call the internal forward logic, not the public one if it has side effects)
            output = self.forward(probe)
            
            # 3. Set scale so output norm becomes 1.0
            measured_norm = output.norm()
            self.scale = 1.0 / measured_norm
        
        # The Theoretical Correction
        # Note: Because of the "Block" splitting implementation required to bypass the 32k limit, 
        # the energy distribution might not be perfectly uniform across the slice boundary. 
        # Method 1 (Auto-Calibration above) is safer because it accounts for the specific block artifacts 
        # of your exact dimensions.
        # self.scale = (1.0 / self.N) * torch.sqrt(self.N / torch.tensor(self.output_dim, device=device))

    def _safe_hadamard(self, x):
        """Splits x into blocks that the CUDA kernel can handle."""
        # x shape: (batch, 65536) -> reshape to (batch * 2, 32768)
        orig_shape = x.shape
        x = x.view(-1, self.block_size)
        x = hadamard_transform(x)
        return x.view(orig_shape)

    def forward(self, v):
        if v.dim() == 1:
            v = v.unsqueeze(0)
        
        # 1. Pad to N (65536)
        v_padded = torch.zeros((v.shape[0], self.N), device=v.device, dtype=v.dtype)
        v_padded[:, :self.input_dim] = v
        
        # 2. Fastfood Pipeline with Block-Hadamard
        # Stage 1: B * H
        x = v_padded * self.B
        x = self._safe_hadamard(x)
        
        # Stage 2: Global Permutation (This mixes the two blocks)
        x = x[:, self.P]
        
        # Stage 3: G * H
        x = x * self.G
        x = self._safe_hadamard(x)
        
        # 3. Scale and Slice
        V = x * self.scale
        return V[:, :self.output_dim]
    