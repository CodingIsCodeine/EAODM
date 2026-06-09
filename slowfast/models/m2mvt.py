"""
M²MVT – Multi-view Multi-modal Vision Transformer
==================================================
Faithful reproduction of the architecture described in:
  "Early Anticipation of Driving Maneuvers" (ECCV 2024)

Architecture summary
--------------------
Multi-view branch (5 views: front, rear, left, right, driver)
  • Each view → independent linear projection (patch embedding) → token sequence
  • All 5 sequences concatenated → z_mv_n
  • Learnable CLS token e_mv_cls prepended → z_mv_0
  • N-2 learnable episodic memory tokens prepended → full MV input

Ego branch (1 view: aria_gaze)
  • View → linear projection → token sequence
  • Learnable CLS token e_ev_cls prepended
  • 2 learnable episodic memory tokens prepended → full EV input

Shared MViT encoder (same encoder, different modalities)
  • z_mv_0 → MViT encoder → e_mv_cls (first output token)
  • z_ev_0 → MViT encoder → e_ev_cls (first output token)

Classification head
  • concat(e_mv_cls, e_ev_cls) → Linear → NUM_CLASSES

Key paper quotes implemented verbatim:
  "Separate projections are used for all five views."
  "Embeddings from multiple views are fused into a combined sequence."
  "A joint CLS token e_mv_cls is learned for this branch."
  "For the ego view, a separate CLS token e_ev_cls is learned."
  "The classification tokens are concatenated before anticipation."
  "For M2MVT, we pass different modalities through the same encoder."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from slowfast.models.video_model_builder import MViT      # existing repo encoder
from slowfast.models.build import MODEL_REGISTRY


# ── Episodic memory: 12 tokens total; 10 for MV, 2 for EV ────────────────────
NUM_MEMORY_TOTAL = 12
NUM_MEMORY_MV    = NUM_MEMORY_TOTAL - 2   # 10
NUM_MEMORY_EV    = 2


@MODEL_REGISTRY.register()
class M2MVT(nn.Module):
    """
    Multi-view Multi-modal Vision Transformer for maneuver anticipation.

    Input during forward():
        mv_list  : list of 5 tensors  [B, C, T, H, W]
                   (front, rear, left, right, driver)
        ego_list : list of 1 tensor   [B, C, T, H, W]
                   (aria_gaze)
        active_mv_views  : optional list of bools len=5 (for single-view ablation)
        active_ego_views : optional list of bools len=1 (for single-view ablation)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── build the shared MViT encoder (re-used for both branches) ─────────
        # We extract the encoder layers from an MViT instance but share weights.
        self.encoder = _build_mvit_encoder(cfg)

        d_model = self.encoder.embed_dim  # 96 (grows through DIM_MUL)

        # Final embedding dim after MViT (after all DIM_MUL: 96 → 192 → 384 → 768)
        d_final = self.encoder.final_dim

        # ── separate per-view linear projections (patch embeddings) ──────────
        # "Separate projections are used for all five views."
        # Each view gets its own Xk projection matrix.
        patch_dim = self._patch_dim(cfg)   # t*h*w*C of one tube

        self.mv_projections = nn.ModuleList([
            nn.Linear(patch_dim, d_model) for _ in range(5)
        ])
        self.ego_projection = nn.Linear(patch_dim, d_model)

        # ── learnable CLS tokens ──────────────────────────────────────────────
        self.e_mv_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.e_ev_cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # ── episodic memory tokens ────────────────────────────────────────────
        self.E_mv_eps = nn.Parameter(torch.zeros(1, NUM_MEMORY_MV, d_model))
        self.E_ev_eps = nn.Parameter(torch.zeros(1, NUM_MEMORY_EV, d_model))

        # ── classification head ───────────────────────────────────────────────
        # concat(e_mv_cls, e_ev_cls) → 2 * d_final → NUM_CLASSES
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_final),
            nn.Dropout(cfg.MODEL.DROPOUT_RATE),
            nn.Linear(2 * d_final, cfg.MODEL.NUM_CLASSES),
        )

        # Weight initialisation
        nn.init.trunc_normal_(self.e_mv_cls, std=0.02)
        nn.init.trunc_normal_(self.e_ev_cls, std=0.02)
        nn.init.trunc_normal_(self.E_mv_eps, std=0.02)
        nn.init.trunc_normal_(self.E_ev_eps, std=0.02)

    # ── helper: patch / tube dimension ───────────────────────────────────────
    def _patch_dim(self, cfg):
        k = cfg.MVIT.PATCH_KERNEL    # (3, 7, 7)
        t, h, w = k
        C = cfg.DATA.INPUT_CHANNEL_NUM[0]
        return int(t * h * w * C)

    # ── patchify one view ─────────────────────────────────────────────────────
    def _patchify(self, x):
        """
        x: [B, C, T, H, W]
        Returns: [B, N, patch_dim] where N = number of non-overlapping tubes
        """
        B, C, T, H, W = x.shape
        k = self.cfg.MVIT.PATCH_KERNEL    # (pt, ph, pw)
        s = self.cfg.MVIT.PATCH_STRIDE    # (st, sh, sw)
        p = self.cfg.MVIT.PATCH_PADDING   # (pp, ph, pw)

        # Use unfold to extract tubes
        x = x.unfold(2, k[0], s[0]).unfold(3, k[1], s[1]).unfold(4, k[2], s[2])
        # x: [B, C, nT, nH, nW, k0, k1, k2]
        nT, nH, nW = x.shape[2], x.shape[3], x.shape[4]
        x = x.contiguous().view(B, C, nT, nH, nW, -1)
        # [B, C, nT, nH, nW, kt*kh*kw]
        x = x.permute(0, 2, 3, 4, 1, 5)
        # [B, nT, nH, nW, C, kt*kh*kw]
        x = x.reshape(B, nT * nH * nW, -1)
        # [B, N, C*kt*kh*kw]
        return x

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, mv_list, ego_list,
                active_mv_views=None, active_ego_views=None):
        """
        mv_list  : list of 5 tensors [B, C, T, H, W]
        ego_list : list of 1 tensor  [B, C, T, H, W]
        active_mv_views  : list of 5 bools; None = all active
        active_ego_views : list of 1 bool;  None = all active
        """
        B = mv_list[0].shape[0]

        if active_mv_views is None:
            active_mv_views = [True] * 5
        if active_ego_views is None:
            active_ego_views = [True] * 1

        # ── Multi-view branch ─────────────────────────────────────────────────
        mv_tokens = []
        for k, (x, proj, active) in enumerate(
            zip(mv_list, self.mv_projections, active_mv_views)
        ):
            if not active:
                # Zero-fill inactive views so the sequence length is preserved
                patches = self._patchify(x)               # [B, N, patch_dim]
                token   = proj(torch.zeros_like(patches)) # [B, N, d_model]
            else:
                patches = self._patchify(x)               # [B, N, patch_dim]
                token   = proj(patches)                    # [B, N, d_model]
            mv_tokens.append(token)

        # Fuse: concatenate along token dimension
        z_mv_n = torch.cat(mv_tokens, dim=1)              # [B, 5*N, d_model]

        # Prepend CLS + episodic memory
        cls_mv  = self.e_mv_cls.expand(B, -1, -1)         # [B, 1, d_model]
        mem_mv  = self.E_mv_eps.expand(B, -1, -1)         # [B, 10, d_model]
        z_mv_0  = torch.cat([cls_mv, mem_mv, z_mv_n], dim=1)  # [B, 1+10+5N, d]

        # Run shared encoder
        e_mv_cls = self.encoder(z_mv_0)                   # [B, d_final]

        # ── Ego branch ────────────────────────────────────────────────────────
        x_ego   = ego_list[0]
        if not active_ego_views[0]:
            patches_ego = self._patchify(x_ego)
            tok_ego = self.ego_projection(torch.zeros_like(patches_ego))
        else:
            patches_ego = self._patchify(x_ego)           # [B, N, patch_dim]
            tok_ego = self.ego_projection(patches_ego)     # [B, N, d_model]

        cls_ev  = self.e_ev_cls.expand(B, -1, -1)         # [B, 1, d_model]
        mem_ev  = self.E_ev_eps.expand(B, -1, -1)         # [B, 2, d_model]
        z_ev_0  = torch.cat([cls_ev, mem_ev, tok_ego], dim=1)  # [B, 1+2+N, d]

        # Reuse same encoder
        e_ev_cls = self.encoder(z_ev_0)                   # [B, d_final]

        # ── Late fusion + classification ──────────────────────────────────────
        # "The classification tokens are concatenated before anticipation."
        z = torch.cat([e_mv_cls, e_ev_cls], dim=1)        # [B, 2 * d_final]
        logits = self.head(z)                              # [B, NUM_CLASSES]

        return logits


# ── Internal: thin MViT encoder wrapper ──────────────────────────────────────

def _build_mvit_encoder(cfg):
    """
    Construct and return the inner MViT encoder as a standalone nn.Module.
    We wrap the existing MViT class to expose only the transformer body
    (no patchification, no final head – M2MVT replaces those).
    """
    return M2MVTEncoder(cfg)


class M2MVTEncoder(nn.Module):
    """
    Wraps the MViT transformer blocks.

    Input:  x [B, L, d_model]   – pre-patchified token sequence (includes CLS)
    Output: cls_token [B, d_final]  – the CLS token after all transformer layers
    """

    def __init__(self, cfg):
        super().__init__()
        from slowfast.models.mvit_model import MViT as MViTInternal
        self._mvit = MViTInternal(cfg)

        # Expose dims for M2MVT to use
        self.embed_dim = cfg.MVIT.EMBED_DIM             # 96
        # Compute final dim after DIM_MUL schedule
        d = cfg.MVIT.EMBED_DIM
        for _, mul in cfg.MVIT.DIM_MUL:
            d = int(d * mul)
        self.final_dim = d                               # 768 for the YAML config

    def forward(self, x):
        """
        x: [B, L, d_model]
        Returns: CLS token [B, d_final]
        """
        # The internal MViT transformer blocks expect the sequence directly
        # We call _mvit.forward_tokens which processes the token sequence
        # and returns all tokens; we extract CLS (index 0).
        out = self._mvit.forward_tokens(x)   # [B, L', d_final]
        return out[:, 0, :]                  # CLS token


# ── Single-view inference helper ─────────────────────────────────────────────

def build_active_view_masks(active_views):
    """
    active_views: list of view name strings, e.g. ["front"] or ["aria_gaze"]
    Returns: (active_mv_views: list[bool] len=5, active_ego_views: list[bool] len=1)
    """
    mv_names  = ["front", "rear", "left", "right", "driver"]
    ego_names = ["aria_gaze"]
    active_mv  = [v in active_views for v in mv_names]
    active_ego = [v in active_views for v in ego_names]
    return active_mv, active_ego
