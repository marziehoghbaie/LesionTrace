# -*- coding: utf-8 -*-
"""The code is from https://github.com/LovreAB17/FERLIV-MARIO/tree/main"""
from torch.nn.init import xavier_uniform_

from models.SimaesRetfound.models_vit import VisionTransformer

"""
Created on Tue May 14 15:54:09 2024

@author: TeresaFinis
"""

import torch
import inspect
import os
import sys
import timm
import torch.nn as nn

# local imports
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, grandparentdir)
sys.path.insert(0, grandgrandparentdir)
from models.SimaesRetfound import models_vit
from utils.utils import load_model, get_model_hash

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CrossTransformer(nn.Module):
    """
  Cross Transformer layer
  Reference: https://github.com/cvpaperchallenge/Describing-and-Localizing-Multiple-Change-with-Transformers/tree/main
  """

    def __init__(self, dropout, d_model=512, n_head=4):
        """
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2, attn_mask=None):
        attn_output, attn_weight = self.attention(input1, input2, input2, attn_mask=attn_mask)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)

        return output


class MCCFormers_D(nn.Module):
    """
  MCCFormers-D
  Reference: https://github.com/cvpaperchallenge/Describing-and-Localizing-Multiple-Change-with-Transformers/tree/main
  """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=2, local_attn_size=None):
        """
    :param feature_dim: dimension of input features
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    :param n_layer: number of layers of transformer layer
    """
        super(MCCFormers_D, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))

        self.projection = nn.Sequential(
            nn.Conv2d(feature_dim, d_model, kernel_size=1),
            nn.Dropout(p=dropout)
        )

        self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])

        self.attn_mask = None

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):
        # img_feat1 (batch_size, feature_dim, h, w)
        batch = img_feat1.size(0)
        feature_dim = img_feat1.size(1)
        w, h = img_feat1.size(2), img_feat1.size(3)

        d_model = self.d_model

        img_feat1 = self.projection(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
        img_feat2 = self.projection(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)

        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1), embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)
        # (h, w, d_model)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,
                                                                                     1)  # (batch, d_model, h, w)

        img_feat1 = img_feat1 + position_embedding  # (batch_size, d_model, h, w)
        img_feat2 = img_feat2 + position_embedding  # (batch_size, d_model, h, w)

        output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
        output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)

        for l in self.transformer:
            output1, output2 = l(output1, output2, attn_mask=self.attn_mask), l(output2, output1,
                                                                                attn_mask=self.attn_mask)

        output = torch.cat([output1, output2], dim=2).permute(1, 2, 0)  # (batch_size, 2*d_model, h*w)

        output = output.view(batch, 2 * d_model, h, w)  # (batch_size, 2*d_model, h, w)

        return output


class TransformerEncoder(nn.Module):
    """
  TransformerEncoder
  """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=2, dim_feedforward=2048, pool='max',
                 use_pe=True):
        """
    :param feature_dim: feature dimension of input dimension
    :param d_model: dimension of input to Transformer
    :param n_head: the number of heads in Transformer
    :param n_layers: the number of layers of Transformer
    :param dim_feedforward: dimension of hidden state
    :param h: height of input image
    :param w: width of input image
    """
        super(TransformerEncoder, self).__init__()

        self.pool = pool
        self.use_pe = use_pe
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Conv2d(feature_dim, d_model, kernel_size=1),
            nn.Dropout(p=dropout)
        )

        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=dim_feedforward, dropout=dropout,
                                                   norm_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, features):
        # features (batch_size, feature_dim, h, w)
        batch = features.size(0)
        w, h = features.size(2), features.size(3)

        features = self.input_proj(features)  # (batch_size, d_model, h, w)

        if self.use_pe:
            pos_w = torch.arange(w, device=device).to(device)
            pos_h = torch.arange(h, device=device).to(device)
            embed_w = self.w_embedding(pos_w)
            embed_h = self.h_embedding(pos_h)
            position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1), embed_h.unsqueeze(1).repeat(1, w, 1)],
                                           dim=-1)  # (h, w, d_model)
            position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,
                                                                                         1)  # (batch, d_model, h, w)

            features = features + position_embedding

        features = features.view(batch, self.d_model, -1)  # (batch, d_model, h*w)
        features = features.permute(2, 0, 1)  # (h*w, batch, d_model)

        features = self.transformer(features)  # (h*w, batch, d_model)

        output = features.permute(1, 0, 2)  # (batch_size, h*w, d_model)

        if self.pool == 'max':
            output = torch.max(output, dim=1)[0]
        elif self.pool == 'min':
            output = torch.mean(output, dim=1)

        return output  # (batch_size, d_model)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features):
        logits = self.fc(features)
        return logits


class Model_LovreAntonioBudimir(nn.Module):

    def __init__(self, dropout=0.4, h=14, w=14, d_model=512, n_head=8, n_layers=2, num_classes=4, use_pe_3=False,
                 local_attn_size=None, check_point_path=None, backbone='vit_large'):
        super(Model_LovreAntonioBudimir, self).__init__()
        if backbone == 'vit_base':
            self.feature_extractor = models_vit.vit_base_patch16(
                num_classes=0,
                drop_path_rate=0.1,
                global_pool=False,
                return_features=True,
                sin_cosine_pe=True)
            feature_dim = 768
        else:
            self.feature_extractor = models_vit.vit_large_patch16(
                num_classes=0,
                drop_path_rate=0.1,
                global_pool=False,
                return_features=True,
                sin_cosine_pe=True)
            feature_dim = 1024
        self.feature_extractor.forward = self.feature_extractor.forward_features
        assert check_point_path is not None
        hash_before = get_model_hash(self.feature_extractor)
        print("Hash before loading: {}".format(hash_before), flush=True)
        self.feature_extractor = load_model(check_point_path, self.feature_extractor, device=device, allow_size_mismatch=True)
        hash_after = get_model_hash(self.feature_extractor)
        print("Hash After loading: {}".format(hash_after), flush=True)

        self.change_encoder = MCCFormers_D(
            feature_dim=feature_dim,
            dropout=dropout,
            h=h, w=w, d_model=d_model, n_head=n_head, n_layers=n_layers, local_attn_size=local_attn_size
        )

        self.classification_encoder = TransformerEncoder(
            feature_dim=2 * d_model,
            dropout=dropout,
            h=h,
            w=w,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            use_pe=use_pe_3
        )

        self.cls_head = ClassificationHead(in_channels=d_model, num_classes=num_classes)
        self.feature_dim = feature_dim
        self.w = w
        self.h = h

    def forward(self, img_t0, img_t1):
        batch = img_t0.size(0)

        _, features_t0 = self.feature_extractor(img_t0)  # (batch, w*h, feature_dim)
        _, features_t1 = self.feature_extractor(img_t1)  # (batch, w*h, feature_dim)

        features_t0 = features_t0.permute(0, 2, 1)
        features_t0 = features_t0.view(batch, self.feature_dim, self.h, self.w)  # (batch, feature_dim, h, w)

        features_t1 = features_t1.permute(0, 2, 1)
        features_t1 = features_t1.view(batch, self.feature_dim, self.h, self.w)  # (batch, feature_dim, h, w)

        features = self.change_encoder(features_t0, features_t1)  # (batch_size, 2*d_model, h, w)
        embed = self.classification_encoder(features)  # (batch_size, d_model)

        logits = self.cls_head(embed)

        outputs = {'logits': logits, 'embed': embed, 'features_t0': features_t0, 'features_t1': features_t1}
        return outputs


if __name__ == "__main__":
    model = Model_LovreAntonioBudimir(backbone='vit_base').cuda()
    img_t0 = torch.rand(1, 3, 224, 224).cuda()
    img_t1 = torch.rand(1, 3, 224, 224).cuda()
    print(model(img_t0, img_t1).keys())
