# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

import math

def get_sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe  # shape: (seq_len, d_model)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, return_features=False, sin_cosine_pe=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.return_features = return_features
        self.sin_cosine_pe = sin_cosine_pe
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm
        if self.sin_cosine_pe:
            seq_len = 197
            d_model = 1024
            del self.pos_embed # sinosoidal pe
            self.pos_embed = get_sinusoidal_positional_encoding(seq_len, d_model)


    def forward_features(self, x, return_features=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.to(x.device)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            o = x[:, 1:, :].mean(dim=1)
            features = self.fc_norm(x)
            outcome = self.fc_norm(o)
        else:
            features = self.norm(x)
            outcome = features[:, 0]
        if self.return_features or return_features:
            return outcome, features[:, 1:, :]

        return outcome




def vit_large_patch16(return_features=False, **kwargs):
    model = VisionTransformer(
        img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), return_features=return_features, **kwargs)
    return model

def vit_base_patch16(return_features=False, **kwargs):
    model = VisionTransformer(
        img_size=224,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), return_features=return_features, **kwargs)
    return model




if __name__ == '__main__':
    model = vit_large_patch16(num_classes=2, return_features=True, sin_cosine_pe=False)
    img = torch.randn(2, 3, 224, 224)
    outcome, features = model.forward_features(img)
    print(outcome.shape, features.shape)