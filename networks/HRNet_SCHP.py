#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   AugmentCE2P.py
@Time    :   8/4/19 3:35 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
# Note here we adopt the InplaceABNSync implementation from https://github.com/mapillary/inplace_abn
# By default, the InplaceABNSync module contains a BatchNorm Layer and a LeakyReLu layer
from modules import InPlaceABNSync

from networks.backbone.Lite_HRNET import LiteHRNet

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

# model settings
model_18_small_cfg = dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),                          #TODO Doubt: What do these mean
                num_branches=(2, 3, 4),                         #These correspond to the parallel multi-resolution branches. So, stage 1 has two parallel branches, stage 2 has three parallel branches, etc.
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )),
            with_head=False,
        )

model_30_large_cfg = dict(
            stem=dict(  
                stem_channels=32,
                out_channels=32,
                expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )),
            with_head=True,
            )

model_30_small_cfg = dict(
            stem=dict(  
                stem_channels=32,
                out_channels=32,
                expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )),
            with_head=True,
            )

model_custom_cfg = dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),                          #TODO Doubt: What do these mean
                num_branches=(2, 3, 4),                         #These correspond to the parallel multi-resolution branches. So, stage 1 has two parallel branches, stage 2 has three parallel branches, etc.
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (128, 256),
                    (128, 256, 512),
                    (128, 256, 512, 1024),
                )),
            with_head=False,
        )

model_cfg = {
    "18_small": model_18_small_cfg,
    "30_small": model_30_small_cfg,
    "30_large": model_30_large_cfg,
    "custom": model_custom_cfg
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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

class HRNet_Lite_Custom(nn.Module):
    def __init__(self, num_classes, model_type="custom"):
        self.inplanes = 128
        super(HRNet_Lite_Custom, self).__init__()
        # self.conv1 = conv3x3(3, 64, stride=2)
        # self.bn1 = BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=False)
        # self.conv2 = conv3x3(64, 64)
        # self.bn2 = BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=False)
        # self.conv3 = conv3x3(64, 128)
        # self.bn3 = BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=False)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # self.layer1 = self._make_layer(block, 64, layers[0])                
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.backbone = LiteHRNet(extra = model_cfg[model_type])

        self.up_sample_1 = nn.Upsample(size=(119, 119), mode='bilinear', align_corners=False)
        self.up_sample_2 = nn.Upsample(size=(119, 119), mode='bilinear', align_corners=False)
        self.up_sample_3 = nn.Upsample(size=(60, 60), mode='bilinear', align_corners=False)
        self.up_sample_4 = nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False)

        self.context_encoding = PSPModule(1024, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        # x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        # x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        # x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        # x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        # x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        # x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        # x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)

        x_back = self.backbone(x)                               #Shapes: (8, 128, 32, 32),
                                                                #        (8, 256, 16, 16),
                                                                #        (8, 512, 8, 8),
                                                                #        (8, 1024, 4, 4)         
        x, x2, x3, x4 = x_back[0], x_back[1], x_back[2], x_back[3]

        x = self.up_sample_1(x)                                 #Shape: (8, 128, 32, 32) -> (8, 128, 119, 119)
        x2 = self.up_sample_2(x2)                               #Shape: (8, 256, 16, 16) -> (8, 256, 119, 119)
        x3 = self.up_sample_3(x3)                               #Shape: (8, 512, 8, 8) -> (8, 512, 60, 60)
        x4 = self.up_sample_4(x4)                               #Shape: (8, 1024, 4, 4) -> (8, 1024, 30, 30)

        # x = self.context_encoding(x5)                       #Shape: (8, 512, 30, 30)
        x = self.context_encoding(x4)                           #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fushion(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]



class Decoder_Module_30_small(nn.Module):
    """
    Parsing Branch Decoder Module.
    """

    def __init__(self, num_classes):
        super(Decoder_Module_30_small, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(80, 40, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(40)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 24, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(24)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 40, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(40),
            nn.Conv2d(40, 40, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(40)
        )

        self.conv4 = nn.Conv2d(40, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x

class HRNet_30_small(nn.Module):
    def __init__(self, num_classes, model_type="30_small"):
        self.inplanes = 128
        super(HRNet_30_small, self).__init__()
        # self.conv1 = conv3x3(3, 64, stride=2)
        # self.bn1 = BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=False)
        # self.conv2 = conv3x3(64, 64)
        # self.bn2 = BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=False)
        # self.conv3 = conv3x3(64, 128)
        # self.bn3 = BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=False)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # self.layer1 = self._make_layer(block, 64, layers[0])                
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.backbone = LiteHRNet(extra = model_cfg[model_type])

        self.up_sample_1 = nn.Upsample(size=(119, 119), mode='bilinear', align_corners=False)
        self.up_sample_2 = nn.Upsample(size=(119, 119), mode='bilinear', align_corners=False)
        self.up_sample_3 = nn.Upsample(size=(60, 60), mode='bilinear', align_corners=False)
        self.up_sample_4 = nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False)

        self.context_encoding = PSPModule(160, 80)

        self.edge = Edge_Module(in_fea = [40, 80, 160], mid_fea = 40, out_fea = 2)
        self.decoder = Decoder_Module_30_small(num_classes)

        self.fushion = nn.Sequential(
            nn.Conv2d(160, 80, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(80),
            nn.Dropout2d(0.1),
            nn.Conv2d(80, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        # x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        # x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        # x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        # x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        # x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        # x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        # x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)

        x_back = self.backbone(x)                               #Shapes: (8, 40, 32, 32),
                                                                #        (8, 40, 16, 16),
                                                                #        (8, 80, 8, 8),
                                                                #        (8, 160, 4, 4)         
        x, x2, x3, x4 = x_back[0], x_back[1], x_back[2], x_back[3]

        x = self.up_sample_1(x)                                 #Shape: (8, 40, 32, 32) -> (8, 40, 119, 119)
        x2 = self.up_sample_2(x2)                               #Shape: (8, 40, 16, 16) -> (8, 40, 119, 119)
        x3 = self.up_sample_3(x3)                               #Shape: (8, 80, 8, 8) -> (8, 80, 60, 60)
        x4 = self.up_sample_4(x4)                               #Shape: (8, 160, 4, 4) -> (8, 160, 30, 30)

        x = self.context_encoding(x4)                           #Shape: (8, 80, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)       #Shapes: (8, 20, 119, 119), (8, 40, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 120, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 160, 119, 119). Here, 160 is (3+1)*40.

        fusion_result = self.fushion(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]



def initialize_pretrained_model_hrnet(model, settings, pretrained="./checkpoints/litehrnet_30_coco_256x192.pth"):
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

    if pretrained is not None:
        saved_state_dict = torch.load(pretrained)["state_dict"]
        filt_state_dict = {}
        for k in saved_state_dict:
            if k.split(".")[0] == "backbone":
                new_k = ".".join(k.split(".")[1:])
            # else:
            #     new_k = k

                filt_state_dict[new_k] = saved_state_dict[k]

        # missing, unexp = model.load_state_dict(filt_state_dict, strict=False)
        missing, unexp = model.load_state_dict(saved_state_dict, strict=False)
        print(f"Missing Keys: {len(missing)}, Unexpected Keys: {len(unexp)}")



def hrnet_schp_custom(num_classes=20, pretrained=None):
    model = HRNet_Lite(num_classes, model_type="custom")
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model_hrnet(model, settings, pretrained)
    return model

def hrnet_schp_30_small(num_classes=20, pretrained="./checkpoints/litehrnet_30_coco_256x192.pth"):
    model = HRNet_30_small(num_classes, model_type="30_small")
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model_hrnet(model, settings, pretrained)
    return model
