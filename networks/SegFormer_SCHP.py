#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from mmseg.models.builder import BACKBONES
# from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import attr
import math
import numpy as np

# Note here we adopt the InplaceABNSync implementation from https://github.com/mapillary/inplace_abn
# By default, the InplaceABNSync module contains a BatchNorm Layer and a LeakyReLu layer
from modules import InPlaceABNSync

from .backbone.SegFormer import Mlp, Attention, Block, OverlapPatchEmbed, DWConv
from .backbone.segformer_utils.segformer_head import MLP
from .backbone.segformer_utils.decode_head import BaseDecodeHead

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

affine_par = True

pretrained_settings = {
    'resnet101': {
        'imagenet': {
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.406, 0.456, 0.485],
            'std': [0.225, 0.224, 0.229],
            'num_classes': 1000
        }
    },
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

## CE2P Modules
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)       #Increase channel to match inplanes. inplanes = expansion * planes, ensured in make_layer.
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle

class Edge_Module(nn.Module):
    """
    Edge Learning Branch
    """

    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2):
        super(Edge_Module, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea

class Decoder_Module(nn.Module):
    """
    Parsing Branch Decoder Module.
    """

    def __init__(self, num_classes):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class Decoder_SegCE2P_Module(nn.Module):
    """
    Parsing Branch Decoder Module.
    """

    def __init__(self, num_classes):
        super(Decoder_SegCE2P_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


## SegFormer Modules
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)      #TODO Changed
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape                 #Shape: (8, _, 4, 4)

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])    #Shape: (8, 768, 4, 4)
        # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)  #Shape: (8, 768, 32, 32)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])    #Shape: (8, 768, 8, 8)
        # _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)  #Shape: (8, 768, 32, 32)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])    #Shape: (8, 768, 16, 16)
        # _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)  #Shape: (8, 768, 32, 32)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])    #Shape: (8, 768, 32, 32)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))                       #Shape: (8, 768, 32, 32)

        x = self.dropout(_c)
        x = self.linear_pred(x)     #Shape: (8, 20, 32, 32)

        return x, _c

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # logger = get_root_logger()        #TODO CHANGED
            logger = None
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

class SegFormer_mit_b4(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.backbone = MixVisionTransformer(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        self.head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # decoder_params=dict(embed_dim=768)
        decoder_params=dict(embed_dim=256)
        )       

        self.edge = Edge_Module(in_fea=[64, 128, 320], mid_fea=256, out_fea=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def forward(self, x):

        x = self.backbone(x)

        #Shapes:
        # x2: (8, 64, 32, 32)
        # x3: (8, 128, 16, 16)
        # x4: (8, 320, 8, 8)
        # x5: (8, 512, 4, 4)
        x2, x3, x4, x5 = x
                                
        
        #Decoding
        parsing_result, parsing_fea = self.head(x)  #Shape: (8, 20, 32, 32), (8, 256, 32, 32)

        #Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)   #Shape: (8, 2, 32, 32), (8, 768, 32, 32)

        #Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)   #Shape: (8, 1024, 32, 32)
        fusion_result = self.fusion(x)                  #Shape: (8, 20, 32, 32)

        return [[parsing_result, fusion_result], [edge_result]]

class SegFormer_CE2P_Head(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.backbone = MixVisionTransformer(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        self.context_encoding = PSPModule(512, 512)
        self.head = Decoder_SegCE2P_Module(num_classes)
        self.edge = Edge_Module(in_fea=[64, 128, 320], mid_fea=256, out_fea=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def forward(self, x):

        x = self.backbone(x)

        #Shapes:
        # x2: (8, 64, 32, 32)
        # x3: (8, 128, 16, 16)
        # x4: (8, 320, 8, 8)
        # x5: (8, 512, 4, 4)
        x2, x3, x4, x5 = x
                                
        
        #Decoding
        x = self.context_encoding(x5)                       #Shape: (8, 512, 4, 4)
        parsing_result, parsing_fea = self.head(x, x2)   #Shape: (8, 20, 32, 32), (8, 256, 32, 32)

        #Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)   #Shape: (8, 2, 32, 32), (8, 768, 32, 32)

        #Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)   #Shape: (8, 1024, 32, 32)
        fusion_result = self.fusion(x)                  #Shape: (8, 20, 32, 32)

        return [[parsing_result, fusion_result], [edge_result]]

class SegFormer_b4_mixed(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.backbone = MixVisionTransformer(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        self.head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # decoder_params=dict(embed_dim=768)
        decoder_params=dict(embed_dim=256)
        )       

        self.context_encoding = PSPModule(512, 512)
        self.edge = Edge_Module(in_fea=[64, 128, 320], mid_fea=256, out_fea=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def forward(self, x):

        x = self.backbone(x)

        #Shapes:
        # x2: (8, 64, 32, 32)
        # x3: (8, 128, 16, 16)
        # x4: (8, 320, 8, 8)
        # x5: (8, 512, 4, 4)
        x2, x3, x4, x5 = x
                                
        #Context encoding for x5
        x5 = self.context_encoding(x5)                  #Shape: (8, 512, 4, 4)
        x = [x2, x3, x4, x5]
        
        #Decoding
        parsing_result, parsing_fea = self.head(x)      #Shape: (8, 20, 32, 32), (8, 256, 32, 32)

        #Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)   #Shape: (8, 2, 32, 32), (8, 768, 32, 32)

        #Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)   #Shape: (8, 1024, 32, 32)
        fusion_result = self.fusion(x)                  #Shape: (8, 20, 32, 32)

        return [[parsing_result, fusion_result], [edge_result]]


def initialize_pretrained_model_segformer(model, settings, pretrained='./models/resnet101-imagenet.pth'):
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

    if pretrained is not None:
        saved_state_dict = torch.load(pretrained)
        renamed_state_dict = {f"backbone.{k}":v for k,v in saved_state_dict.items() if not k.__contains__("head")}
        incompatible, unexpected = model.load_state_dict(renamed_state_dict, strict=False)

        incomp_modules = np.unique([p.split(".")[0] for p in incompatible])
        unexp_modules = np.unique([p.split(".")[0] for p in unexpected])
        print(f"\nPre-Trained weights loaded for Backbone!")
        print(f"Pre-Trained Weights not loaded for \n - Incompatile Modules : {incomp_modules}\n - Unexcepted Modules: {unexp_modules}\n")


def segformer_mit_b4(num_classes=20, pretrained="./pretrain_model/mit_b4.pth"):
    model = SegFormer_mit_b4(num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model_segformer(model, settings, pretrained)
    return model

def segformer_b4_ce2p(num_classes=20, pretrained="./pretrain_model/mit_b4.pth"):
    model = SegFormer_CE2P_Head(num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model_segformer(model, settings, pretrained)
    return model

def segformer_b4_mixed(num_classes=20, pretrained="./pretrain_model/mit_b4.pth"):
    model = SegFormer_b4_mixed(num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model_segformer(model, settings, pretrained)
    return model
