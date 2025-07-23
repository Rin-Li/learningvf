from torch import nn, Tensor
import torch
import math 

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        """dim should be a even number"""
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        x: should be (B, 1) -> (B, 1, dim)
        or
        x: should be (B,) -> (B, dim)

        [:,None] is use to unsequence
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb