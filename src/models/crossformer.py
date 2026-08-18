"""
CTG-Crossformer: faithful implementation of the Crossformer architecture
(Zhang & Yan, "Crossformer: Transformer Utilizing Cross-Dimension
Dependency for Multivariate Time Series Forecasting", ICLR 2023), adapted
for binary classification of a 2-variate (FHR, UC) CTG window.

This is architecturally distinct from the pre-existing repo's
`ctg_crossformer.py` (a CNN + bidirectional cross-attention hybrid loosely
named "Crossformer" but not implementing DSW embedding / two-stage
attention / hierarchical merging). PROTOCOL.md section 4 requires the
*published* Crossformer description specifically:

    1. Dimension-Segment-Wise (DSW) embedding
    2. Two-Stage Attention (TSA): cross-time MSA, then cross-dimension
       attention via a small set of learnable "router" tokens (O(D) instead
       of O(D^2) complexity)
    3. Hierarchical encoder: segment-merging between stages, building a
       coarser temporal resolution at each stage (a "pyramid")

Per PROTOCOL.md decision #6, the missingness mask is fused into the DSW
segment embedding (each segment carries both signal values and their mask)
rather than treated as a third attended "dimension" -- cross-dimension
attention only ever runs over D=2 real variates (FHR, UC).
"""
import math
from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Dimension-Segment-Wise (DSW) Embedding
# ---------------------------------------------------------------------------

class DSWEmbedding(nn.Module):
    """
    Splits a (B, T, D) signal (+ (B, T) shared missingness mask) into
    non-overlapping segments of length `seg_len` along T, and linearly
    embeds each (signal-segment, mask-segment) pair into d_model.

    Output: (B, D, L, d_model) where L = T // seg_len.
    """
    def __init__(self, seg_len: int, d_model: int, n_dims: int, n_segments: int):
        super().__init__()
        self.seg_len = seg_len
        self.n_dims = n_dims
        self.n_segments = n_segments
        # Input per segment: seg_len signal values + seg_len mask values.
        self.proj = nn.Linear(seg_len * 2, d_model)
        # Learnable position embedding, per (dimension, segment-position).
        self.pos_embed = nn.Parameter(torch.zeros(1, n_dims, n_segments, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) signal.
            mask: (B, T) float, 1.0 = missing, 0.0 = valid (shared across D).
        Returns:
            (B, D, L, d_model)
        """
        B, T, D = x.shape
        L = T // self.seg_len
        x = x[:, : L * self.seg_len, :]
        mask = mask[:, : L * self.seg_len]

        # (B, L, seg_len, D) -> (B, D, L, seg_len)
        x_seg = x.reshape(B, L, self.seg_len, D).permute(0, 3, 1, 2)
        # mask shared across D -> broadcast to (B, D, L, seg_len)
        mask_seg = mask.reshape(B, L, self.seg_len).unsqueeze(1).expand(B, D, L, self.seg_len)

        combined = torch.cat([x_seg, mask_seg], dim=-1)  # (B, D, L, 2*seg_len)
        embedded = self.proj(combined)  # (B, D, L, d_model)
        embedded = embedded + self.pos_embed[:, :, :L, :]
        return embedded


# ---------------------------------------------------------------------------
# 2. Two-Stage Attention (TSA)
# ---------------------------------------------------------------------------

class CrossTimeAttention(nn.Module):
    """Per-dimension self-attention across the L segment positions."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L, d_model) -> fold D into batch for per-dimension attention over L
        B, D, L, d = x.shape
        x_flat = x.reshape(B * D, L, d)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))
        x_flat = self.norm2(x_flat + self.dropout(self.ff(x_flat)))
        return x_flat.reshape(B, D, L, d)


class CrossDimensionRouterAttention(nn.Module):
    """
    Per-segment-position cross-dimension attention via a small set of
    learnable router tokens: dimensions -> routers (aggregate), then
    routers -> dimensions (distribute). O(D) instead of O(D^2) -- with D=2
    here the complexity saving is moot, but the router mechanism is kept
    faithful to the published architecture (and generalizes if a future
    phase adds more variates).
    """
    def __init__(self, d_model: int, n_heads: int, n_routers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_routers = n_routers
        self.routers = nn.Parameter(torch.zeros(1, 1, n_routers, d_model))
        nn.init.trunc_normal_(self.routers, std=0.02)

        self.agg_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dist_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L, d_model) -> fold L into batch for per-segment-position attention over D
        B, D, L, d = x.shape
        x_perm = x.permute(0, 2, 1, 3).reshape(B * L, D, d)  # (B*L, D, d)

        routers = self.routers.expand(B, L, self.n_routers, d).reshape(B * L, self.n_routers, d)

        # Stage A: routers aggregate from all dimensions (routers = query).
        agg_out, _ = self.agg_attn(routers, x_perm, x_perm)  # (B*L, n_routers, d)

        # Stage B: dimensions read back from the (updated) routers.
        dist_out, _ = self.dist_attn(x_perm, agg_out, agg_out)  # (B*L, D, d)

        x_perm = self.norm1(x_perm + self.dropout(dist_out))
        x_perm = self.norm2(x_perm)
        x_perm = self.norm3(x_perm + self.dropout(self.ff(x_perm)))

        return x_perm.reshape(B, L, D, d).permute(0, 2, 1, 3)  # (B, D, L, d)


class TwoStageAttentionLayer(nn.Module):
    """One TSA block: cross-time attention, then cross-dimension router attention."""
    def __init__(self, d_model: int, n_heads: int, n_routers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_time = CrossTimeAttention(d_model, n_heads, dropout)
        self.cross_dim = CrossDimensionRouterAttention(d_model, n_heads, n_routers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cross_time(x)
        x = self.cross_dim(x)
        return x


# ---------------------------------------------------------------------------
# 3. Hierarchical Encoder (segment merging between stages)
# ---------------------------------------------------------------------------

class SegmentMerge(nn.Module):
    """Merges `merge_factor` adjacent segments (per dimension) into one,
    halving (or /merge_factor) the segment count and doubling the model dim
    internally before projecting back to d_model -- the pyramid-building step."""
    def __init__(self, d_model: int, merge_factor: int = 2):
        super().__init__()
        self.merge_factor = merge_factor
        self.proj = nn.Linear(d_model * merge_factor, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, L, d = x.shape
        m = self.merge_factor
        L_new = L // m
        x = x[:, :, : L_new * m, :].reshape(B, D, L_new, m * d)
        x = self.norm(self.proj(x))
        return x  # (B, D, L_new, d)


class CrossformerEncoderStage(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_tsa_layers: int, n_routers: int,
                 dropout: float, merge: bool, merge_factor: int = 2):
        super().__init__()
        self.merge = SegmentMerge(d_model, merge_factor) if merge else None
        self.layers = nn.ModuleList([
            TwoStageAttentionLayer(d_model, n_heads, n_routers, dropout)
            for _ in range(n_tsa_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge is not None:
            x = self.merge(x)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CTGCrossformer(nn.Module):
    """
    Full CTG-Crossformer: DSW embedding -> hierarchical stack of TSA stages
    (with segment merging between stages) -> pooled classification head.

    Input: x (B, T, 2) [FHR, UC], mask (B, T) shared missingness mask.
    Output: (B, 1) binary distress logit.

    Also exposes `encode()` returning the pre-head pooled latent, for later
    phases that need a shared representation for auxiliary heads / fusion
    (M-A through M-D infusion mechanisms, Phase 3).
    """
    def __init__(
        self,
        seq_len: int = 1800,
        n_dims: int = 2,
        seg_len: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_routers: int = 4,
        n_stages: int = 3,
        n_tsa_layers_per_stage: int = 1,
        merge_factor: int = 2,
        dropout: float = 0.1,
        head_hidden: int = 64,
    ):
        super().__init__()
        assert seq_len % seg_len == 0, "seq_len must be divisible by seg_len"
        n_segments = seq_len // seg_len
        assert n_segments % (merge_factor ** (n_stages - 1)) == 0, (
            f"n_segments={n_segments} must be divisible by merge_factor^(n_stages-1) "
            f"={merge_factor ** (n_stages - 1)} for clean hierarchical merging."
        )

        self.embedding = DSWEmbedding(seg_len, d_model, n_dims, n_segments)

        stages: List[CrossformerEncoderStage] = []
        for stage_i in range(n_stages):
            stages.append(CrossformerEncoderStage(
                d_model=d_model, n_heads=n_heads, n_tsa_layers=n_tsa_layers_per_stage,
                n_routers=n_routers, dropout=dropout,
                merge=(stage_i > 0), merge_factor=merge_factor,
            ))
        self.stages = nn.ModuleList(stages)

        self.latent_dim = d_model
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns the pooled (B, d_model) latent -- mean over final-stage
        segments and dimensions."""
        h = self.embedding(x, mask)  # (B, D, L, d_model)
        for stage in self.stages:
            h = stage(h)
        z = h.mean(dim=(1, 2))  # (B, d_model)
        return z

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, mask)
        return self.head(z)
