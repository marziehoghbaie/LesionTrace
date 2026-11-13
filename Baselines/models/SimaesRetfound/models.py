# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:54:09 2024

@author: TeresaFinis
"""


import inspect
import os
import sys

import torch
import torch.optim as optim

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, grandparentdir)
sys.path.insert(0, grandgrandparentdir)
import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect
import os
import sys

import torch
import torch.optim as optim

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, grandparentdir)
sys.path.insert(0, grandgrandparentdir)
from MARIO_Paper_baselines.models.SimaesRetfound.models_vit import vit_large_patch16, vit_base_patch16
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


class SiameseNetwork(nn.Module):

    def __init__(self, nb_classes, device='cpu', check_point_path=None, backbone="vit_large"):
        super(SiameseNetwork, self).__init__()
        if backbone == "vit_large":
            self.model = vit_large_patch16(
                num_classes=nb_classes,
                drop_path_rate=0.1,
                global_pool=True,
                return_features=True)
        elif backbone == "vit_base":
            self.model = vit_base_patch16(
                num_classes=nb_classes,
                drop_path_rate=0.1,
                global_pool=True,
                return_features=True)
        # hash_before = get_model_hash(self.model)
        # print("Hash before loading: {}".format(hash_before), flush=True)
        # self.model = load_model(check_point_path, self.model, device=device, allow_size_mismatch=True)
        # hash_after = get_model_hash(self.model)
        # print("Hash After loading: {}".format(hash_after), flush=True)
        # print('[INFO] Model is loaded from {}'.format(check_point_path))

        self.fc_in_features = self.model.head.in_features

        self.model.forward = self.model.forward_features

        # todo you should call the ustils here

        f = self.model.forward_features
        ff = self.model.patch_embed
        fff = self.model.pos_embed

        del self.model.head

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
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
        output, features = self.model(x)

        output = output.view(output.size()[0], -1)
        return output, features

    def forward(self, input1, input2):

        output1, features_t0 = self.forward_once(input1)
        output2, features_t1 = self.forward_once(input2)

        embed = torch.cat((output1, output2), 1)
        self.features = embed
        logits = self.fc(embed)
        outputs = {'logits': logits, 'embed': embed, 'features_t0': features_t0, 'features_t1': features_t1}
        return outputs


if __name__ == '__main__':
    device = torch.device('cpu')
    model = SiameseNetwork(nb_classes=2, device=device, backbone="vit_large")
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)
    outputs = model(img1, img2)
    print(outputs.keys())

