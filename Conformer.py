"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import random
import datetime
import time
import datetime
import scipy.io

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, input_ch=18):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (input_ch, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor, agnostic=False) -> Tensor:
        if agnostic:
            out = self.shallownet[:2](x)
            return out
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class channel_selection(nn.Module):
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.masks = nn.Parameter(torch.ones(num_channels))
        self.indexes = nn.Parameter(torch.ones(num_channels))
    
    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (B, num_patches + 1, dim). 
        """
        output = input_tensor.mul(self.indexes)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        self.select1 = channel_selection(emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        q, k, v = self.queries(x), self.keys(x), self.values(x)
        q = self.select1(q)
        queries = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = self.select1(k)
        keys = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = self.select1(v)
        values = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__()
        # nn.Linear(emb_size, expansion * emb_size),
        # nn.GELU(),
        # nn.Dropout(drop_p),
        # nn.Linear(expansion * emb_size, emb_size),
        self.net1 = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p)
        )
        self.net2 = nn.Linear(expansion * emb_size, emb_size)
        self.select2 = channel_selection(expansion * emb_size)
    
    def forward(self, x):
        # pruning   
        x = self.net1(x)
        # pruning  
        x = self.select2(x)
        x = self.net2(x)
        return x


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(34760, 256),
        #     nn.ELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 32),
        #     nn.ELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32, 4)
        # )

    def forward(self, x):
        # x = x.contiguous().view(x.size(0), -1)
        # out = self.fc(x)
        out = self.clshead(x)
        return out, self.clshead[0](x)


class Conformer(nn.Sequential):
    def __init__(self, emb_size=400, depth=6, n_classes=2, input_ch=18, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, input_ch),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    print(time.asctime(time.localtime(time.time())))