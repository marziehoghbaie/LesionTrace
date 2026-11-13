"""
LesionChangeDetector — v1

Architecture:
  - Shared ViT-B/16 backbone (RetFiner pretrained)
  - Shared lesion scorer: patch tokens → map1, map2  (B, 196)
  - Lesion-masked cross-attention: compare patch content inside lesion regions
    → inside_feat  (B, 32)  same area, different content
    → extent_feat  (B, 32)  lesion appeared or resolved
  - CLS cross-attention: global temporal context
    → cls_delta    (B, 768) no projection
  - Fusion classifier: cat([inside_feat, extent_feat, cls_delta]) → (B, 832) → 3 classes

Supervised by:
  - FocalLoss on logits
  - Dice+BCE on map1, map2 (per-visit lesion GT)
  - DirectionalAngleLoss on cls_delta
"""
"""
LesionChangeDetector — v2
Changes vs v1
-------------
* LesionMaskedCrossAttention.forward() now returns:
    inside_feat, extent_feat, lesion_attn, inter_mask, extent_mask, residual
* CLSCrossAttention.forward() unchanged (already returns attn weights).
* LesionChangeDetector.forward() output dict gains:
    'inter_mask'   (B, 196)  — min(map1_sig, map2_sig)
    'extent_mask'  (B, 196)  — max - min, clamped
    'residual'     (B, 196, 768)  — p2_lesion - attn_out
    'map1_sig'     (B, 196)  — sigmoid lesion scores T1 (flat)
    'map2_sig'     (B, 196)  — sigmoid lesion scores T2 (flat)
  All existing keys are preserved unchanged.
"""

import math
import logging
from functools import partial
from copy import deepcopy
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
from timm.models.registry import register_model
from utils import *
from torch.utils.tensorboard import SummaryWriter
from test_files.test_utils import scoring, save_results

_logger = logging.getLogger(__name__)


# ─── Config helper ────────────────────────────────────────────────────────────

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_class': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0'
            '--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
}


# ─── ViT backbone blocks ──────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = head_dim ** -0.5
        self.qkv        = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
                   .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x    = self.proj(x)
        x    = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1      = norm_layer(dim)
        self.attn       = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2      = norm_layer(dim)
        mlp_hidden_dim  = int(dim * mlp_ratio)
        self.mlp        = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop)

    def forward(self, x):
        attn_out, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


# ─── 1. Shared lesion scorer ──────────────────────────────────────────────────

class LesionScorer(nn.Module):
    """
    Shared across T1 and T2.
    patch_tokens (B, 196, 768) → per-patch lesion logits (B, 196).
    Bias init = -2.75 (conservative: most patches healthy).
    """
    def __init__(self, embed_dim=768, bottleneck=64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.constant_(self.scorer[-1].bias, -2.75)
        nn.init.xavier_uniform_(self.scorer[-1].weight)

    def forward(self, patch_tokens):
        return self.scorer(patch_tokens).squeeze(-1)   # (B, 196)


# ─── 2. Lesion-masked cross-attention ─────────────────────────────────────────

class LesionMaskedCrossAttention(nn.Module):
    """
    Compares patch content inside lesion regions between T1 and T2.

    Returns
    -------
    inside_feat  (B, out_dim)
    extent_feat  (B, out_dim)
    lesion_attn  (B, num_heads, 196, 196)
    inter_mask   (B, 196)   min(map1_sig, map2_sig)   ← NEW
    extent_mask  (B, 196)   max - min, ≥ 0            ← NEW
    residual     (B, 196, 768)  p2_lesion - attn_out  ← NEW
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1, out_dim=32):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)

        self.inside_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        self.extent_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        for proj in [self.inside_proj, self.extent_proj]:
            nn.init.xavier_uniform_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

    def forward(self, p1, p2, map1_sig, map2_sig):
        # ── soft masking ──────────────────────────────────────────────────────
        p1_lesion = p1 * map1_sig.unsqueeze(-1)   # (B, 196, 768)
        p2_lesion = p2 * map2_sig.unsqueeze(-1)   # (B, 196, 768)

        # ── cross-attention ───────────────────────────────────────────────────
        attn_out, lesion_attn = self.cross_attn(
            query=self.norm_q(p2_lesion),
            key=self.norm_k(p1_lesion),
            value=p1_lesion,
            need_weights=True,
            average_attn_weights=False,   # (B, heads, 196, 196)
        )

        # ── residual: what changed at each patch ──────────────────────────────
        residual = p2_lesion - attn_out            # (B, 196, 768)  ← NEW

        # ── spatial masks ─────────────────────────────────────────────────────
        inter_mask  = torch.min(map1_sig, map2_sig)                         # (B, 196) ← NEW
        extent_mask = (torch.max(map1_sig, map2_sig) - inter_mask           # (B, 196) ← NEW
                       ).clamp(min=0.0)
        eps = 1e-6

        # ── weighted aggregation ──────────────────────────────────────────────
        inside_vec = (inter_mask.unsqueeze(-1) * residual).sum(1) / (
            inter_mask.sum(1, keepdim=True) + eps)
        extent_vec = (extent_mask.unsqueeze(-1) * residual).sum(1) / (
            extent_mask.sum(1, keepdim=True) + eps)

        inside_feat = self.inside_proj(inside_vec)   # (B, out_dim)
        extent_feat = self.extent_proj(extent_vec)   # (B, out_dim)

        return inside_feat, extent_feat, lesion_attn, inter_mask, extent_mask, residual


# ─── 3. CLS cross-attention ───────────────────────────────────────────────────

class CLSCrossAttention(nn.Module):
    """
    Lightweight cross-attention for global temporal context.

    Returns
    -------
    cls_delta      (B, 768)
    cls2_mixed     (B, 768)
    cls1_mixed     (B, 768)
    cls_attn_1to2  (B, num_heads, 1, 196)
    cls_attn_2to1  (B, num_heads, 1, 196)
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn_1_to_2 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_2_to_1 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.norm_p1 = nn.LayerNorm(embed_dim)
        self.norm_p2 = nn.LayerNorm(embed_dim)

    def forward(self, cls1, cls2, p1, p2):
        cls1_ctx, cls_attn_1to2 = self.attn_1_to_2(
            query=self.norm1(cls1.unsqueeze(1)),
            key=self.norm_p2(p2),
            value=p2,
            need_weights=True,
            average_attn_weights=False,
        )
        cls1_mixed = cls1 + cls1_ctx.squeeze(1)

        cls2_ctx, cls_attn_2to1 = self.attn_2_to_1(
            query=self.norm2(cls2.unsqueeze(1)),
            key=self.norm_p1(p1),
            value=p1,
            need_weights=True,
            average_attn_weights=False,
        )
        cls2_mixed = cls2 + cls2_ctx.squeeze(1)

        cls_delta = cls2_mixed - cls1_mixed
        return cls_delta, cls2_mixed, cls1_mixed, cls_attn_1to2, cls_attn_2to1


# ─── Main model ───────────────────────────────────────────────────────────────

_TRAINABLE_MODULE_NAMES = (
    'lesion_scorer',
    'lesion_cross_attn',
    'cls_cross_attn',
    'classifier',
)


class LesionChangeDetector(nn.Module):
    """
    Retinal OCT structural change detector.

    Output dict keys
    ----------------
    Existing (unchanged):
        logits, map1, map2, map1_raw, map2_raw,
        cls_delta, cls1_mixed, cls2_mixed, final_embd,
        lesion_attn, cls_attn_1to2, cls_attn_2to1

    New:
        map1_sig     (B, 196)       sigmoid lesion scores T1 (flat, pre-upsample)
        map2_sig     (B, 196)       sigmoid lesion scores T2
        inter_mask   (B, 196)       min(map1_sig, map2_sig)
        extent_mask  (B, 196)       max(map1_sig, map2_sig) - min, ≥ 0
        residual     (B, 196, 768)  p2_lesion - cross_attn_out
    """

    def __init__(
        self,
        img_size        = 224,
        patch_size      = 16,
        in_chans        = 3,
        num_classes     = 3,
        embed_dim       = 768,
        depth           = 12,
        num_heads       = 12,
        mlp_ratio       = 4.,
        qkv_bias        = True,
        drop_rate       = 0.,
        attn_drop_rate  = 0.,
        drop_path_rate  = 0.1,
        embed_layer     = PatchEmbed,
        norm_layer      = None,
        act_layer       = None,
        weight_init     = '',
        scorer_bottleneck = 64,
        local_heads     = 8,
        local_out_dim   = 32,
        cls_heads       = 8,
        head_dropout    = 0.1,
    ):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer  = act_layer  or nn.GELU

        self.num_classes  = num_classes
        self.embed_dim    = embed_dim
        self.num_features = embed_dim

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches   = self.patch_embed.num_patches
        self.map_sdim = int(sqrt(num_patches))   # 14

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.lesion_scorer = LesionScorer(
            embed_dim=embed_dim, bottleneck=scorer_bottleneck)

        self.lesion_cross_attn = LesionMaskedCrossAttention(
            embed_dim=embed_dim, num_heads=local_heads,
            dropout=0.1, out_dim=local_out_dim)

        self.cls_cross_attn = CLSCrossAttention(
            embed_dim=embed_dim, num_heads=cls_heads, dropout=0.1)

        fusion_dim = local_out_dim * 2 + embed_dim   # 832
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, num_classes),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.init_weights(weight_init)
        nn.init.constant_(self.lesion_scorer.scorer[-1].bias, -2.75)
        nn.init.xavier_uniform_(self.lesion_scorer.scorer[-1].weight)

    # ── weight init ───────────────────────────────────────────────────────────

    def init_weights(self, mode=''):
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # ── backbone ──────────────────────────────────────────────────────────────

    def forward_features(self, x):
        x          = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x          = torch.cat((cls_tokens, x), dim=1)
        x          = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]   # cls (B,D), patch_tokens (B,N,D)

    # ── main forward ──────────────────────────────────────────────────────────

    def forward(self, x1, x2):
        B, _, H, W = x1.shape

        # step 1 — backbone
        cls1, p1 = self.forward_features(x1)
        cls2, p2 = self.forward_features(x2)

        # step 2 — shared lesion scorer
        map1_logits = self.lesion_scorer(p1)       # (B, 196) logits
        map2_logits = self.lesion_scorer(p2)
        map1_sig    = torch.sigmoid(map1_logits)   # (B, 196) ∈ [0,1]
        map2_sig    = torch.sigmoid(map2_logits)

        # step 3 — local pathway (now returns 6 values)
        inside_feat, extent_feat, lesion_attn, \
            inter_mask, extent_mask, residual = self.lesion_cross_attn(
                p1, p2, map1_sig, map2_sig)

        # step 4 — global pathway
        cls_delta, cls2_mixed, cls1_mixed, \
            cls_attn_1to2, cls_attn_2to1 = self.cls_cross_attn(
                cls1, cls2, p1, p2)

        # step 5 — fusion + classify
        feat   = torch.cat([inside_feat, extent_feat, cls_delta], dim=1)  # (B,832)
        logits = self.classifier(feat)

        # step 6 — upsample maps for loss / vis
        map1_spatial = map1_sig.view(B, 1, self.map_sdim, self.map_sdim)
        map2_spatial = map2_sig.view(B, 1, self.map_sdim, self.map_sdim)
        map1_up = F.interpolate(map1_spatial, size=(H, W),
                                mode='bilinear', align_corners=False)[:, 0]
        map2_up = F.interpolate(map2_spatial, size=(H, W),
                                mode='bilinear', align_corners=False)[:, 0]

        return {
            # ── existing keys (unchanged) ─────────────────────────────────────
            'logits':        logits,
            'map1':          map1_up,        # (B, H, W)
            'map2':          map2_up,        # (B, H, W)
            'map1_raw':      map1_sig,       # (B, 196)  kept for loss compat
            'map2_raw':      map2_sig,       # (B, 196)
            'cls_delta':     cls_delta,      # (B, 768)
            'cls1_mixed':    cls1_mixed,     # (B, 768)
            'cls2_mixed':    cls2_mixed,     # (B, 768)
            'final_embd':    feat,           # (B, 832)
            'lesion_attn':   lesion_attn,    # (B, heads, 196, 196)
            'cls_attn_1to2': cls_attn_1to2,  # (B, heads, 1, 196)
            'cls_attn_2to1': cls_attn_2to1,  # (B, heads, 1, 196)
            # ── new keys ──────────────────────────────────────────────────────
            'map1_sig':      map1_sig,       # (B, 196)  alias (same tensor)
            'map2_sig':      map2_sig,       # (B, 196)
            'inter_mask':    inter_mask,     # (B, 196)
            'extent_mask':   extent_mask,    # (B, 196)
            'residual':      residual,       # (B, 196, 768)
        }

# ─── Weight init helper ───────────────────────────────────────────────────────

def _init_vit_weights(module, name='', head_bias=0., jax_impl=False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# ─── Checkpoint loading ───────────────────────────────────────────────────────
# identical to your existing _load_weights — loads ViT backbone only,
# new modules (lesion_scorer, lesion_cross_attn, cls_cross_attn, classifier)
# are randomly initialized and trained from scratch

@torch.no_grad()
def _load_weights(model: LesionChangeDetector,
                  checkpoint_path: str,
                  prefix: str = ''):
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:   w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3: w = w.transpose([2, 0, 1])
            elif w.ndim == 2: w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    model.patch_embed.proj.weight.copy_(
        adapt_input_conv(
            model.patch_embed.proj.weight.shape[1],
            _n2p(w[f'{prefix}embedding/kernel'])
        )
    )
    model.patch_embed.proj.bias.copy_(
        _n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(
        _n2p(w[f'{prefix}cls'], t=False))

    pos_embed_w = _n2p(
        w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(
            pos_embed_w, model.pos_embed,
            1, model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)

    model.norm.weight.copy_(
        _n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(
        _n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    for i, block in enumerate(model.blocks.children()):
        bp  = f'{prefix}Transformer/encoderblock_{i}/'
        mha = bp + 'MultiHeadDotProductAttention_1/'

        block.norm1.weight.copy_(_n2p(w[f'{bp}LayerNorm_0/scale']))
        block.norm1.bias.copy_(  _n2p(w[f'{bp}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha}{n}/kernel'], t=False).flatten(1).T
            for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha}{n}/bias'], t=False).reshape(-1)
            for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(
            _n2p(w[f'{mha}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(
            _n2p(w[f'{mha}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r+1}').weight.copy_(
                _n2p(w[f'{bp}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r+1}').bias.copy_(
                _n2p(w[f'{bp}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{bp}LayerNorm_2/scale']))
        block.norm2.bias.copy_(  _n2p(w[f'{bp}LayerNorm_2/bias']))

    _logger.info('[INFO] ViT backbone loaded. New modules randomly initialised.')


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    _logger.info('Resized pos embed: %s → %s',
                 posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok  = posemb[:, :num_tokens]
        posemb_grid = posemb[0, num_tokens:]
        ntok_new   -= num_tokens
    else:
        posemb_tok  = posemb[:, :0]
        posemb_grid = posemb[0]
    gs_old      = int(math.sqrt(len(posemb_grid)))
    gs_new      = [int(math.sqrt(ntok_new))] * 2 if not len(gs_new) else gs_new
    posemb_grid = (posemb_grid
                   .reshape(1, gs_old, gs_old, -1)
                   .permute(0, 3, 1, 2))
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = (posemb_grid
                   .permute(0, 2, 3, 1)
                   .reshape(1, gs_new[0] * gs_new[1], -1))
    return torch.cat([posemb_tok, posemb_grid], dim=1)


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(
                v, model.pos_embed, 1,
                model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768,
        depth=12, num_heads=12, **kwargs)
    return build_model_with_cfg(
        LesionChangeDetector,
        'vit_base_patch16_224',
        pretrained,
        default_cfg          = default_cfgs['vit_base_patch16_224'],
        pretrained_filter_fn = checkpoint_filter_fn,
        **model_kwargs,
    )


# ─── Loss functions ───────────────────────────────────────────────────────────

def loss_fn_dice_bce(pred, target, sigmoid=True, smooth=1.0, bce_weight=0.5):
    if sigmoid:
        pred_sig = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred.flatten(1), target.flatten(1).float(), reduction='mean'
        )
    else:
        pred_sig = pred
        # clamp to avoid log(0) — safe under autocast
        pred_clamped = pred_sig.flatten(1).clamp(1e-7, 1 - 1e-7)
        bce_loss = F.binary_cross_entropy(
            pred_clamped,
            target.flatten(1).float(),
            reduction='mean'
        )

    pred_flat = pred_sig.flatten(1)
    tgt_flat  = target.flatten(1).float()

    intersection = (pred_flat * tgt_flat).sum(dim=1)
    dice_loss    = (1.0 - (2.0 * intersection + smooth) /
                   (pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) + smooth)).mean()

    return (1.0 - bce_weight) * dice_loss + bce_weight * bce_loss


# ─── Backbone freeze / unfreeze ───────────────────────────────────────────────

def set_backbone_frozen(model: LesionChangeDetector, frozen: bool):
    for name, param in model.named_parameters():
        is_new = any(name.startswith(m) for m in _TRAINABLE_MODULE_NAMES)
        param.requires_grad = True if is_new else (not frozen)
    status = 'FROZEN' if frozen else 'UNFROZEN'
    _logger.info(f'[Stage] Backbone {status}. '
                 f'Trainable: {_TRAINABLE_MODULE_NAMES}')


# ─── Validation loop ──────────────────────────────────────────────────────────

def test(loader, model, loss_fn, logger, train_config,
         direction_loss_fn=None, complete=False, save_path=None, model_stats=False):

    model.eval()
    device = next(model.parameters()).device

    y_pred, y_true, y_probs, case_ids = [], [], [], []
    loss_terms_epoch = []
    total_loss = 0.0
    embedds = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for idx, batch in enumerate(loader):
            if idx % 100 == 0:
                logger.info(f'[INFO] Evaluating batch {idx}/{len(loader)}')

            A      = batch['A'].to(device)
            B      = batch['B'].to(device)
            labels = batch['label'].to(device)
            if model_stats:
                stats_dict = utils.get_model_stats(
                    model,
                    A,
                    B
                )

                logger.info('Model type: {}, # Parameters: {} M(illion), FLOPS: {} G'.format(train_config['model_type'], stats_dict['params_m'], stats_dict['flops_g']))
                return
            outputs = model(A, B)
            logits  = outputs['logits']
            probs   = F.softmax(logits, dim=1)
            preds   = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            case_ids.extend(batch['name'])
            if complete:
                if 'prototypes_strength' in outputs:
                    embedds.extend(
                        outputs['prototypes_strength'].detach().cpu().numpy()
                    )
                else:
                    assert len(A) == 1, "complete=True requires batch_size=1"
                    embedds.append(outputs['final_embd'][0].detach().cpu().numpy())
                continue
            # ── loss computation ──────────────────────────────────────────────
            cls_loss   = loss_fn(logits, labels)
            loss_total = train_config.get('cls_loss_coef', 1.0) * cls_loss
            loss_terms = {'classification': cls_loss.item()}

            if direction_loss_fn is not None:
                dir_loss   = direction_loss_fn(outputs['cls1_mixed'], outputs['cls2_mixed'], labels)
                loss_total = loss_total + \
                    train_config.get('DirectionalAngleLosscoeff', 0.3) * dir_loss
                loss_terms['direction_loss'] = dir_loss.item()

            if train_config.get('use_map_loss', False):
                maskA = batch['maskA_binary'].float().to(device)
                maskB = batch['maskB_binary'].float().to(device)

                mask_changed = (labels != 1)
                mask_stable  = (labels == 1)
                mask_has_lesion = (
                    (maskA.sum(dim=[-1,-2]) > 0) |
                    (maskB.sum(dim=[-1,-2]) > 0)
                )
                map_loss = torch.tensor(0.0, device=device)

                valid_changed = mask_changed & mask_has_lesion
                if valid_changed.sum() > 0:
                    map_loss = map_loss + (
                        loss_fn_dice_bce(
                            outputs['map1'][valid_changed],
                            maskA[valid_changed], sigmoid=True
                        ) +
                        loss_fn_dice_bce(
                            outputs['map2'][valid_changed],
                            maskB[valid_changed], sigmoid=True
                        )
                    ) / 2.0

                valid_stable = mask_stable & mask_has_lesion
                if valid_stable.sum() > 0:
                    stable_coef = train_config.get('stable_map_coef', 0.3)
                    map_loss = map_loss + stable_coef * (
                        loss_fn_dice_bce(
                            outputs['map1'][valid_stable],
                            maskA[valid_stable], sigmoid=True
                        ) +
                        loss_fn_dice_bce(
                            outputs['map2'][valid_stable],
                            maskB[valid_stable], sigmoid=True
                        )
                    ) / 2.0

                loss_total = loss_total + \
                    train_config.get('lambda_mse', 1.0) * map_loss
                loss_terms['map_loss'] = map_loss.item()

            total_loss += loss_total.item() * labels.size(0)
            loss_terms_epoch.append(loss_terms)


    avg_loss = total_loss / len(loader.dataset)
    bacc     = balanced_accuracy_score(y_true, y_pred)
    scores   = scoring(y_true, y_pred)
    F1_score = scores['F1_score']

    logger.info(
        f'[INFO] Validation — F1={F1_score:.4f} | '
        f'BAcc={bacc:.4f} | Loss={avg_loss:.4f}')
    logger.info(f'[INFO] Full scores: {scores}')
    logger.info('-' * 60)

    rps = sa_rps_metric(
        torch.tensor(y_probs), torch.tensor(y_true),
        len(loader.dataset.classes)
    )
    logger.info(f'[INFO] RPS (ordinality): {rps}')
    if complete and save_path is not None:
        save_results(
            y_probs, y_pred, y_true, case_ids, save_path, embedds,
            loader.dataset.classes.keys(), postfix="test_complete", logger=logger
        )
    return {
        'F1_score':          F1_score,
        'bacc':              bacc,
        'loss':              avg_loss,
        'loss_terms_summary': loss_terms_epoch,
        'rps_metric':        rps,
        'mean_scores':       sum(scores.values()) / len(scores),
    }

def test_external(loader, model, logger, save_path=None):
    from sklearn.manifold import TSNE

    model.eval()
    device = next(model.parameters()).device

    y_pred = []
    y_change_magnitude = []
    y_probs = []
    case_ids = []
    embeddings = []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for idx, batch in enumerate(loader):
            if idx % 100 == 0:
                logger.info(f'[INFO] Evaluating batch {idx}/{len(loader)}')

            A = batch['A'].to(device)
            B = batch['B'].to(device)
            change_magnitude = batch['change_magnitude']

            outputs = model(A, B)

            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_change_magnitude.extend(change_magnitude.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            case_ids.extend(batch['name'])

            # assumes batch size = 1 (as in your original code)
            assert A.size(0) == 1, "This implementation requires batch_size=1"
            embeddings.append(outputs['final_embd'][0].detach().cpu().numpy())

    embeddings = np.array(embeddings)

    if len(embeddings) < 2:
        logger.warning("Not enough samples for t-SNE.")
        return

    perplexity = min(30, len(embeddings) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        init='random',
        random_state=42
    )

    tsne_result = tsne.fit_transform(embeddings)

    df = pd.DataFrame(tsne_result, columns=['tsne1', 'tsne2'])
    df['change_magnitude'] = y_change_magnitude
    df['pred'] = y_pred
    df['case_id'] = case_ids

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, 'tsne_embeddings.csv'), index=False)

    return df
# ─── Training loop ────────────────────────────────────────────────────────────

def train_val(train_config, logger, model, optimizer, loss_fn,
              train_loader, val_loader, scheduler,
              ckpt_saver, init_epoch=0, args=None):
    logger.info('[INFO] Training Lesion Change Detector ... ')
    writer = SummaryWriter(log_dir=ckpt_saver.save_dir)
    device = torch.device(
        'cuda:0' if torch.cuda.is_available()
                    and train_config['use_gpu'] else 'cpu')
    logger.info(f'[INFO] Device: {device}')
    model = model.to(device)

    scaler         = torch.cuda.amp.GradScaler()
    grad_clip_norm = train_config.get('grad_clip_norm', 1.0)
    freeze         = train_config.get('freeze', False)
    freeze_epochs  = train_config.get('freeze_backbone_epochs', 0)
    use_ema        = train_config.get('use_ema', False)
    train_config['save_metric'] = 'f1'

    if use_ema:
        from copy import deepcopy
        ema = ModelEMA(model, decay=train_config.get('ema_decay', 0.999))
        logger.info('[INFO] EMA enabled.')

    # ── DirectionalAngleLoss ──────────────────────────────────────────────────
    # NOTE: your existing DirectionalAngleLoss takes (cls1, cls2, labels).
    # With the new model we pass cls_delta directly.
    # You will need to adapt DirectionalAngleLoss to accept a single
    # delta vector instead of two separate CLS tokens, or compute it
    # as delta = cls2 - cls1 inside the loss. Flag this for review.
    direction_loss_fn = None
    if train_config.get('DirectionalAngleLoss', False):
        direction_loss_fn = DirectionalAngleLoss(
            feature_dim = model.embed_dim,
            logger      = logger,
            args        = args,
        ).to(device)
        optimizer.add_param_group({
            'params':       list(direction_loss_fn.parameters()),
            'lr':           train_config.get('base_lr', 1e-4),
            'weight_decay': 0.0,
            'name':         'direction_loss_fn',
        })

    best_performance    = 0.0
    save_bacc           = 0.5
    log_interval        = 20
    best_checkpoint_path = None
    epochs_acc, epochs_loss       = [], []
    epochs_acc_val, epochs_loss_val = [], []

    if freeze:
        set_backbone_frozen(model, frozen=True)
        logger.info(f'[INFO] Stage 1: backbone frozen for {freeze_epochs} epochs.')
    else:
        set_backbone_frozen(model, frozen=False)
        logger.info('[INFO] Nothing is frozen.')

    for epoch in range(init_epoch, train_config['num_epochs']):

        # ── stage transition ──────────────────────────────────────────────────
        if epoch == init_epoch + freeze_epochs and freeze:
            set_backbone_frozen(model, frozen=False)
            for pg in optimizer.param_groups:
                if not any(m in pg.get('name','')
                           for m in _TRAINABLE_MODULE_NAMES):
                    pg['lr'] *= 0.1
            logger.info('[INFO] Stage 2: backbone unfrozen, LR ×0.1.')

        model.train()
        head_lr = [pg['lr'] for pg in optimizer.param_groups
                   if 'classifier' in pg.get('name', '')]
        logger.info(
            f'\n[INFO] Epoch {epoch} | LR (head): {head_lr[:1]} | '
            f'Stage: {"1-frozen" if epoch < init_epoch+freeze_epochs else "2-unfrozen"}'
        )

        running_loss  = 0.0
        y_true, y_pred = [], []
        all_losses     = []
        epoch_start    = time()

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            A      = batch['A'].to(device)
            B      = batch['B'].to(device)
            labels = batch['label'].to(device)
            maskA  = batch['maskA_binary'].float().to(device)
            maskB  = batch['maskB_binary'].float().to(device)

            # optional temporal flip augmentation
            if train_config.get('temporal_flip_aug', False):
                flip = torch.rand(A.shape[0]) > 0.5
                if flip.any():
                    A_f, B_f = A[flip].clone(), B[flip].clone()
                    A[flip], B[flip] = B_f, A_f
                    labels[flip] = 2 - labels[flip]
                    mA_f, mB_f = maskA[flip].clone(), maskB[flip].clone()
                    maskA[flip], maskB[flip] = mB_f, mA_f

            loss_terms = {}

            with torch.cuda.amp.autocast(enabled=True):

                outputs = model(A, B)
                logits  = outputs['logits']
                preds   = F.softmax(logits, dim=1).argmax(dim=1)

                # classification loss
                cls_loss   = loss_fn(logits, labels)
                loss_total = train_config.get('cls_loss_coef', 1.0) * cls_loss
                loss_terms['classification'] = cls_loss.item()

                # directional loss
                if direction_loss_fn is not None:
                    dir_loss   = direction_loss_fn(outputs['cls1_mixed'], outputs['cls2_mixed'], labels)
                    loss_total = loss_total + \
                        train_config.get('DirectionalAngleLosscoeff', 0.3) * dir_loss
                    loss_terms['direction'] = dir_loss.item()

                # map loss
                if train_config.get('use_map_loss', False):
                    mask_changed    = (labels != 1)
                    mask_stable     = (labels == 1)
                    mask_has_lesion = (
                        (maskA.sum(dim=[-1,-2]) > 0) |
                        (maskB.sum(dim=[-1,-2]) > 0)
                    )
                    map_loss = torch.tensor(0.0, device=device)

                    valid_changed = mask_changed & mask_has_lesion
                    if valid_changed.sum() > 0:
                        map_loss = map_loss + (
                            loss_fn_dice_bce(
                                outputs['map1'][valid_changed],
                                maskA[valid_changed], sigmoid=True
                            ) +
                            loss_fn_dice_bce(
                                outputs['map2'][valid_changed],
                                maskB[valid_changed], sigmoid=True
                            )
                        ) / 2.0

                    valid_stable = mask_stable & mask_has_lesion
                    if valid_stable.sum() > 0:
                        stable_coef = train_config.get('stable_map_coef', 0.3)
                        map_loss = map_loss + stable_coef * (
                            loss_fn_dice_bce(
                                outputs['map1'][valid_stable],
                                maskA[valid_stable], sigmoid=True
                            ) +
                            loss_fn_dice_bce(
                                outputs['map2'][valid_stable],
                                maskB[valid_stable], sigmoid=True
                            )
                        ) / 2.0

                    loss_total = loss_total + \
                        train_config.get('lambda_mse', 1.0) * map_loss
                    loss_terms['map_changed'] = map_loss.item()

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            if use_ema:
                ema.update(model)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            running_loss += loss_total.item() * labels.size(0)
            all_losses.append(loss_terms)

            if idx % log_interval == 0:
                bacc   = balanced_accuracy_score(y_true, y_pred)
                scores = scoring(y_true, y_pred)
                logger.info(
                    f'  [Epoch {epoch}][Batch {idx}] '
                    f'loss={loss_total.item():.4f} '
                    f'bacc={bacc:.4f} '
                    f'F1={scores["F1_score"]:.4f}'
                )

        logger.info(
            f'[INFO] Epoch {epoch} took {(time()-epoch_start)/60:.1f} min.')
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = balanced_accuracy_score(y_true, y_pred)
        train_sc   = scoring(y_true, y_pred)
        logger.info(
            f'[TRAIN] Epoch {epoch} | Loss={epoch_loss:.4f} | '
            f'BACC={epoch_acc:.4f} | F1={train_sc["F1_score"]:.4f}')
        logger.info(
            f'        Loss breakdown: '
            f'{pd.DataFrame(all_losses).mean().to_dict()}')

        if use_ema:
            ema.apply_shadow(model)

        val_start   = time()
        val_metrics = test(
            val_loader, model, loss_fn, logger,
            train_config      = train_config,
            direction_loss_fn = direction_loss_fn,
        )
        logger.info(
            f'[INFO] Validation took {(time()-val_start)/60:.1f} min.')

        if use_ema:
            ema.restore(model)

        val_perf = (val_metrics['bacc']
                    if train_config['save_metric'] == 'bacc'
                    else val_metrics['F1_score'])
        val_loss = val_metrics['loss']

        if 'loss_terms_summary' in val_metrics:
            mean_val = pd.DataFrame(
                val_metrics['loss_terms_summary']).mean()
            logger.info(f'[VAL]   Loss breakdown: {mean_val.to_dict()}')
        logger.info(
            f'[VAL]   Per-class scores: {val_metrics["mean_scores"]}')
        logger.info(
            f'[VAL]   Epoch {epoch} | Loss={val_loss:.4f} | '
            f'F1={val_perf:.4f}')
        logger.info('-----------------------------------------------')

        if val_perf > best_performance:
            best_performance = val_perf
            if best_performance > save_bacc:
                save_dict = {
                    'epoch_idx':          epoch,
                    'model_state_dict':   model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if use_ema:
                    save_dict['ema_state_dict'] = ema.state_dict()
                best_checkpoint_path = ckpt_saver.save(
                    save_dict, val_perf,
                    metric=train_config['save_metric'])
                logger.info(
                    f'[CKPT]  Saved at epoch {epoch} | '
                    f'f1={val_perf:.4f}')

        if scheduler is not None:
            if train_config.get('scheduler') == 'reduce_on_plateau':
                scheduler.step(val_perf)
            else:
                scheduler.step()

        writer.add_scalar('ACC/train', epoch_acc,  epoch)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Perf/val',   val_perf,   epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)

        epochs_acc.append(epoch_acc)
        epochs_loss.append(epoch_loss)
        epochs_acc_val.append(val_perf)
        epochs_loss_val.append(val_loss)

    writer.close()
    draw_results(
        [epochs_loss, epochs_acc, epochs_loss_val, epochs_acc_val],
        save_path=ckpt_saver.save_dir)
    return model, best_checkpoint_path




def build_optimizer_with_layer_decay(
        model: LesionChangeDetector,
        base_lr: float,
        weight_decay: float,
        layer_decay: float = 0.75):

    depth          = len(model.blocks)
    no_decay_names = {'pos_embed', 'cls_token', 'bias'}  # removed dist_token
    param_groups   = []

    def _wd(name):
        return 0.0 if any(nd in name for nd in no_decay_names) else weight_decay

    lowest_lr = base_lr * (layer_decay ** depth)

    # patch embedding — lowest LR
    for name, param in model.patch_embed.named_parameters():
        if param.requires_grad:
            param_groups.append({
                'params': [param], 'lr': lowest_lr,
                'weight_decay': _wd(name), 'name': f'patch_embed.{name}',
            })

    # cls_token and pos_embed — lowest LR, no weight decay
    for pname in ('cls_token', 'pos_embed'):
        p = getattr(model, pname, None)
        if p is not None and p.requires_grad:
            param_groups.append({
                'params': [p], 'lr': lowest_lr,
                'weight_decay': 0.0, 'name': pname,
            })

    # ViT blocks — layer-wise LR decay
    # block 0 gets lowest_lr, block 11 gets base_lr * layer_decay^1
    for i, block in enumerate(model.blocks):
        lr_i = base_lr * (layer_decay ** (depth - i))
        for name, param in block.named_parameters():
            if param.requires_grad:
                param_groups.append({
                    'params': [param], 'lr': lr_i,
                    'weight_decay': _wd(name), 'name': f'block_{i}.{name}',
                })

    # final norm — one step above lowest
    for name, param in model.norm.named_parameters():
        if param.requires_grad:
            param_groups.append({
                'params': [param],
                'lr': base_lr * (layer_decay ** 1),
                'weight_decay': _wd(name), 'name': f'norm.{name}',
            })

    # new modules — full base_lr
    for module_name in _TRAINABLE_MODULE_NAMES:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for name, param in module.named_parameters():
            if param.requires_grad:
                param_groups.append({
                    'params': [param], 'lr': base_lr,
                    'weight_decay': _wd(name),
                    'name': f'{module_name}.{name}',
                })

    return torch.optim.AdamW(param_groups)







# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('LesionChangeDetector smoke test')
    print('=' * 60)

    model = LesionChangeDetector(
        img_size=224, patch_size=16, in_chans=3, num_classes=3)

    n_total    = sum(p.numel() for p in model.parameters()) / 1e6
    n_new      = sum(
        p.numel() for n, p in model.named_parameters()
        if any(n.startswith(m) for m in _TRAINABLE_MODULE_NAMES)
    ) / 1e6
    n_backbone = n_total - n_new

    print(f'Total params:    {n_total:.1f}M')
    print(f'Backbone params: {n_backbone:.1f}M')
    print(f'New modules:     {n_new:.3f}M')

    x = torch.randn(2, 3, 224, 224)
    out = model(x, x)

    assert out['logits'].shape   == (2, 3),       f"logits: {out['logits'].shape}"
    assert out['map1'].shape     == (2, 224, 224), f"map1: {out['map1'].shape}"
    assert out['map2'].shape     == (2, 224, 224), f"map2: {out['map2'].shape}"
    assert out['map1_raw'].shape == (2, 196),      f"map1_raw: {out['map1_raw'].shape}"
    assert out['cls_delta'].shape == (2, 768),     f"cls_delta: {out['cls_delta'].shape}"

    print(f'logits:    {out["logits"].shape}   ✓')
    print(f'map1:      {out["map1"].shape} ✓')
    print(f'map1_raw:  {out["map1_raw"].shape}    ✓')
    print(f'cls_delta: {out["cls_delta"].shape}   ✓')

    # scorer bias check
    bias = model.lesion_scorer.scorer[-1].bias.item()
    assert abs(bias - (-2.75)) < 0.01, f'scorer bias: {bias}'
    print(f'scorer bias: {bias:.3f} (expected -2.75) ✓')

    print('=' * 60)
    print('All checks passed.')
    print('=' * 60)