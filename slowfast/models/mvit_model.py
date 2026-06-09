"""
MViT internal model exposing `forward_tokens` for M²MVT.

This file provides M2MVTEncoder's internal MViT with a `forward_tokens` method
that accepts a pre-patchified token sequence [B, L, D] and returns
all output tokens [B, L', D_final] after all transformer blocks.

The architecture follows MViTv2-S (Small) as configured by the YAML:
  DEPTH = 16, EMBED_DIM = 96, NUM_HEADS = 1, DIM_MUL at layers [1,3,14]

We implement this from scratch (no dependency on the SlowFast MViT class)
so that the encoder can accept an arbitrary-length token sequence from M²MVT
(which already handles patchification externally).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MViTBlock(nn.Module):
    """
    One MViT transformer block with pooling attention.
    Simplified for token-sequence inputs (spatial/temporal dims treated as 1D).
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        dim_mul_in_att=True,
        residual_pooling=True,
        # pool params (simplified: identity pool for 1D token streams)
        q_stride=1,
    ):
        super().__init__()
        self.dim_in  = dim_in
        self.dim_out = dim_out
        self.dim_mul_in_att = dim_mul_in_att
        self.residual_pooling = residual_pooling
        self.num_heads = num_heads
        self.head_dim  = max(dim_out // num_heads, 1)

        # Determine attention dim
        if dim_mul_in_att:
            att_dim = dim_out
        else:
            att_dim = dim_in

        self.norm1 = norm_layer(dim_in)
        self.norm2 = norm_layer(att_dim if dim_mul_in_att else dim_in)

        # QKV
        self.q = nn.Linear(dim_in,   num_heads * self.head_dim, bias=qkv_bias)
        self.k = nn.Linear(dim_in,   num_heads * self.head_dim, bias=qkv_bias)
        self.v = nn.Linear(dim_in,   num_heads * self.head_dim, bias=qkv_bias)

        self.proj    = nn.Linear(num_heads * self.head_dim, att_dim)
        self.proj_kv = nn.Linear(dim_in, att_dim)  # residual projection for dim change

        self.mlp = nn.Sequential(
            nn.Linear(att_dim if dim_mul_in_att else dim_in,
                      int((att_dim if dim_mul_in_att else dim_in) * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int((att_dim if dim_mul_in_att else dim_in) * mlp_ratio), dim_out),
        )

        # Linear to bring residual to dim_out when dim changes
        self.proj_res = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: [B, L, dim_in]
        B, L, _ = x.shape

        # ── attention ─────────────────────────────────────────────────────────
        normed = self.norm1(x)
        q = self.q(normed).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(normed).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(normed).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale
        attn  = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        out = self.proj(out)       # [B, L, att_dim or dim_in]

        # Residual with optional dim projection
        if self.dim_mul_in_att:
            x_res = self.proj_res(x)  # [B, L, dim_out] ... but att_dim may != dim_out
            # att_dim == dim_out when dim_mul_in_att, so x_res has same dim as out
        else:
            x_res = x               # [B, L, dim_in]

        x2 = x_res + self.drop_path(out)

        # ── MLP ───────────────────────────────────────────────────────────────
        normed2 = self.norm2(x2)
        mlp_out = self.mlp(normed2)  # → [B, L, dim_out]

        # Residual
        dim_out = mlp_out.shape[-1]
        if x2.shape[-1] != dim_out:
            if hasattr(self.proj_res, 'weight'):
                x2 = self.proj_res(x2)
            else:
                x2 = F.linear(x2, torch.eye(dim_out, x2.shape[-1], device=x2.device))
        x3 = x2 + self.drop_path(mlp_out)
        return x3


class DropPath(nn.Module):
    """Stochastic depth for residual connections."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device).ge_(1 - keep).div(keep)
        return x * mask


class MViT(nn.Module):
    """
    MViTv2-S transformer body.
    Receives a pre-tokenised sequence [B, L, D] and outputs all tokens [B, L, D_final].
    The CLS token (index 0) is used for classification.

    Configuration is read directly from cfg.MVIT.*
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        depth      = cfg.MVIT.DEPTH         # 16
        embed_dim  = cfg.MVIT.EMBED_DIM     # 96
        num_heads  = cfg.MVIT.NUM_HEADS     # 1
        mlp_ratio  = cfg.MVIT.MLP_RATIO     # 4.0
        qkv_bias   = cfg.MVIT.QKV_BIAS      # True
        drop_path  = cfg.MVIT.DROPPATH_RATE  # 0.2
        norm_layer = nn.LayerNorm

        # DIM_MUL: [(layer_idx, multiplier), ...]
        dim_mul_map = dict(cfg.MVIT.DIM_MUL)   # {1: 2.0, 3: 2.0, 14: 2.0}

        # Build blocks
        blocks = []
        dim_in  = embed_dim
        for i in range(depth):
            mul    = dim_mul_map.get(i, 1.0)
            dim_out = int(dim_in * mul)

            dpr = drop_path * i / (depth - 1)   # linear drop path schedule

            blocks.append(
                MViTBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr,
                    norm_layer=norm_layer,
                    dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                    residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                )
            )
            dim_in = dim_out

        self.blocks   = nn.ModuleList(blocks)
        self.norm     = norm_layer(dim_in)
        self.final_dim = dim_in   # 768 after 3× DIM_MUL of 2.0

    def forward_tokens(self, x):
        """
        x: [B, L, D]  – token sequence with CLS prepended
        Returns all output tokens [B, L, D_final].
        """
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
