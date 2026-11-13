# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import timm.models.vision_transformer
from timm.models.layers import PatchEmbed
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import load_model, get_model_hash


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, return_features=False, patch_embeddings=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.return_features = return_features
        if patch_embeddings is not None:
            self.patch_embeddings = patch_embeddings
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            features = self.norm(x)
            outcome = features[:, 0]
        if self.return_features:
            return outcome, features

        return outcome


def vit_large_patch16(return_features=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,in_chans=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), return_features=return_features, patch_embeddings=PatchEmbed(in_chans=2),
        **kwargs)
    return model


def vit_base_patch16(return_features=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,in_chans=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), return_features=return_features, patch_embeddings=PatchEmbed(in_chans=2),
        **kwargs)
    return model


class EarlySiameseNetwork(nn.Module):

    def __init__(self, nb_classes, device='cpu', check_point_path=None, backbone="vit_large"):
        super(EarlySiameseNetwork, self).__init__()
        if backbone == "vit_large":
            self.model = vit_large_patch16(
                num_classes=nb_classes,
                drop_path_rate=0.1,
                global_pool=True)
        elif backbone == "vit_base":
            self.model = vit_base_patch16(
                num_classes=nb_classes,
                drop_path_rate=0.1,
                global_pool=True)
        hash_before = get_model_hash(self.model)
        print("Hash before loading: {}".format(hash_before), flush=True)
        self.model = load_model(check_point_path, self.model, device=device, allow_size_mismatch=True)
        hash_after = get_model_hash(self.model)
        print("Hash After loading: {}".format(hash_after), flush=True)
        print('[INFO] Model is loaded from {}'.format(check_point_path))

        self.fc_in_features = self.model.head.in_features

        self.model.forward = self.model.forward_features

        # todo you should call the ustils here

        f = self.model.forward_features
        ff = self.model.patch_embed
        fff = self.model.pos_embed

        del self.model.head

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, nb_classes),
        )

        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.model(x)

        output = output.view(output.size()[0], -1)

        return output

    def forward(self, input1, input2):

        inputs = torch.cat((input1, input2), 1)
        embed = self.forward_once(inputs)

        self.features = embed
        logits = self.fc(embed)
        outputs = {'logits': logits, 'embed': embed}
        return outputs

