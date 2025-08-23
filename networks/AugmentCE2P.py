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

class Fusion_Conv_Attention(nn.Module):
    def __init__(self, num_classes, C=1024, reduction=64, k_spatial=5, spatial_groups=4):
        """
        C              : Channels of input
        reduction      : reduction factor for channel-attention MLP
        k_spatial      : kernel size for spatial gating conv
        spatial_groups : how many channel-groups to gate independently
        """
        super().__init__()
        G = spatial_groups
        assert C % G == 0, "total channels must be divisible by spatial_groups"


        #Grouped spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(
            in_channels=C,
            out_channels=G,
            kernel_size=k_spatial,
            padding=k_spatial//2,
            groups=G,
            bias=True
        ),
        nn.Sigmoid()
        )

        #Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # → (B, C, 1, 1)
            nn.Conv2d(C, C//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C//reduction, C, 1, bias=False),
            nn.Sigmoid()
        )

        #Prepare output
        C_mid = (C + num_classes)//2
        self.prep_out = nn.Sequential(
            nn.Conv2d(C, C_mid, kernel_size=1, bias=False),
            InPlaceABNSync(C_mid),
            nn.Dropout2d(0.1),
            nn.Conv2d(C_mid, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):

        #Channel attention
        ca = self.ca(x)                     #Shape: (B, C, 1, 1)
        x_ca = x * ca                       #Shape: (B, C, H, W)

        #Grouped Spatial Atention
        gates = self.sa(x_ca)               #Shape: (B, G, H, W)

        #Align gates with input shape to get weighted input
        B, C, H, W = x_ca.shape
        G = gates.shape[1]
        C_per_G = C // G

        x_g = x_ca.view(B, G, C_per_G, H, W)                                    #Shape: (B, C, H, W) -> (B, G, C_per_G, H, W)
        gates_g = gates.view(B, G, 1, H, W).expand(-1, -1, C_per_G, -1, -1)     #Shape: (B, G, H, W) -> (B, G, 1, H, W) -> (B, G, C_per_G, H, W)
        x_sa = (x_g * gates_g).reshape(B, C, H, W)                              #Shape: (B, G, C_per_G, H, W) -> (B, C, H, W)

        # x_mix = x + x_sa                    #Residual Connection

        #Prepare the output
        out = self.prep_out(x_sa)           #Shape: (B, num_classes, H, W)
        return out

class Fusion_Conv_Block(nn.Module):
    def __init__(self, num_classes, C=1024, reduction=64, k_spatial=5, spatial_groups=4):
        """
        C              : Channels of input
        reduction      : reduction factor for channel-attention MLP
        k_spatial      : kernel size for spatial gating conv
        spatial_groups : how many channel-groups to gate independently
        """
        super().__init__()
        G = spatial_groups
        self.C = C
        assert C % G == 0, "total channels must be divisible by spatial_groups"


        #Grouped spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(
            in_channels=C,
            out_channels=G,
            kernel_size=k_spatial,
            padding=k_spatial//2,
            groups=G,
            bias=True
        ),
        nn.Sigmoid()
        )

        #Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # → (B, C, 1, 1)
            nn.Conv2d(C, C//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C//reduction, C, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        #Channel attention
        ca = self.ca(x)                     #Shape: (B, C, 1, 1)
        x_ca = x * ca                       #Shape: (B, C, H, W)

        #Grouped Spatial Atention
        gates = self.sa(x_ca)               #Shape: (B, G, H, W)

        #Align gates with input shape to get weighted input
        B, C, H, W = x_ca.shape
        G = gates.shape[1]
        C_per_G = C // G

        x_g = x_ca.view(B, G, C_per_G, H, W)                                    #Shape: (B, C, H, W) -> (B, G, C_per_G, H, W)
        gates_g = gates.view(B, G, 1, H, W).expand(-1, -1, C_per_G, -1, -1)     #Shape: (B, G, H, W) -> (B, G, 1, H, W) -> (B, G, C_per_G, H, W)
        x_sa = (x_g * gates_g).reshape(B, C, H, W)                              #Shape: (B, G, C_per_G, H, W) -> (B, C, H, W)

        x_mix = x + x_sa                    #Residual Connection
        return F.relu(x_mix, inplace=True)

class Fusion_Cross_Attention(nn.Module):
    def __init__(self, num_classes, c_p=256, heads=4):
        super().__init__()
        self.heads = heads
        self.c_p_down = c_p // 4
        self.dk = (self.c_p_down // heads) ** -0.5

        #Project input to lower dims
        self.conv_channel_down = nn.Conv2d(4*c_p, c_p, 1, bias=False, groups=4)
        self.conv_spatial_down = nn.Conv2d(c_p, c_p, 3, stride=2, padding=1, bias=False)
        

        #Projections for query, key and value
        self.to_q = nn.Conv2d(self.c_p_down, self.c_p_down, 1, bias=False)
        self.to_k = nn.Conv2d(self.c_p_down, self.c_p_down, 1, bias=False)
        self.to_v = nn.Conv2d(self.c_p_down, self.c_p_down, 1, bias=False)
        self.proj = nn.Conv2d(self.c_p_down, self.c_p_down, 1, bias=True)

        #Attention to Output
        self.prep_out = nn.Sequential(
            nn.Conv2d(4 * self.c_p_down, 9 * num_classes, 1, bias=False),
            InPlaceABNSync(9 * num_classes),
            nn.Dropout2d(0.1),
            nn.Conv2d(9 * num_classes, 3 * num_classes, 1, bias=False),
            InPlaceABNSync(3 * num_classes),
            nn.Dropout2d(0.1),
            nn.Conv2d(3 * num_classes, num_classes, 1, bias=True)
        )

    def forward(self, x):
        """
        P: (B, c_p, H, W)  — parsing features
        E: (B, n_edges*c_p, H, W)  — concat of edge features
        """

        #Downsample the input to make attention feasible
        x_down = self.conv_channel_down(x)                      #Shape: (B, 1024, H, W) -> (B, 256, H, W)
        #x_down = self.conv_spatial_down(x_down)                 #Shape: (B, 256, )
        #x_down = x
    
        #Split the input into edge and parsing features
        feats = x_down.chunk(4, dim=1)                          #Input is made of four sub-inputs
        edge_feats, parsing_feat = feats[:3], feats[-1]         #Shapes: (B, 192, H, W), (B, 64, H, W)
        B, c_p_down, H, W = parsing_feat.shape

        # 2) project parsing → K,V and reshape for multi-head
        k = self.to_k(parsing_feat).view(B, self.heads, self.c_p_down//self.heads, H*W)  # (B, h, d_k, N)
        v = self.to_v(parsing_feat).view(B, self.heads, self.c_p_down//self.heads, H*W)  # (B, h, d_k, N)

        # 3) for each edge‐scale, do Q→Attention→out
        attended = []
        for e in edge_feats:
            # Q from this edge
            q = self.to_q(e).view(B, self.heads, self.c_p_down//self.heads, H*W)  # (B, h, d_k, N)

            # compute scaled dot-prod attention
            # attn   : (B, h, N_q, N_k) = softmax( qᵀ·k  * scale )
            attn = torch.einsum('bhcn,bhkn->bhck', q, k) * self.dk
            attn = attn.softmax(dim=-1)

            # out_i  : (B, h, d_k, N_q) = attn · v
            out_i = torch.einsum('bhck,bhkn->bhcn', attn, v)
            attended.append(out_i)
        
        attended_reshaped = [o.reshape(B, self.c_p_down, H, W) for o in attended]   #Shape: 3 * (B, 64, H, W)
        attended_reshaped.append(parsing_feat)
        attended_agg = torch.cat(attended_reshaped, dim=1)                  #Shape: (B, 256, H, W)

        #Prepare output
        output = self.prep_out(attended_agg)              #Shape: (B, 20, H, W)

        return output

class MultiScale_Module(nn.Module):
    def __init__(self, in_channels=2048, 
                 kernel_sizes=(1, 3, 5, 7)):
        """
        Args:
            in_channels:  number of channels in (B, C, H, W)
            kernel_sizes: iterable of odd ints, e.g. (1,3,5,7)
        """
        super().__init__()
        assert all(k % 2 == 1 for k in kernel_sizes), "kernel_sizes must be odd"

        # one conv per scale, each preserves C channels and spatial size
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, padding=k//2, bias=False, groups=in_channels),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                InPlaceABNSync(in_channels),
                nn.Dropout2d(0.1)
            )
            for k in kernel_sizes
        ])
        # fuse back to original C channels
        self.project = nn.Sequential(
            nn.Conv2d(
            in_channels * len(kernel_sizes),
            in_channels,
            kernel_size=1,
            bias=False
        ),
        InPlaceABNSync(in_channels),
        nn.Dropout2d(0.1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        feats = [conv(x) for conv in self.branches]       #Shape: Each is (B, C, H, W)
        cat  = torch.cat(feats, dim=1)                    #Shape: (B, C*scales, H, W)
        out  = x + self.project(cat)                      #Shape: (B, C, H, W)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)

        
        x = self.context_encoding(x5)                       #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fushion(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]

class ResNet_Fusion_Conv_Attention(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet_Fusion_Conv_Attention, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fusion = Fusion_Conv_Attention(num_classes)

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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)
        x = self.context_encoding(x5)                       #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fusion(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]

class ResNet_Fusion_Conv_Blocks(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet_Fusion_Conv_Blocks, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fusion_1 = Fusion_Conv_Block(num_classes)
        self.fusion_2 = Fusion_Conv_Block(num_classes)
        self.fusion_3 = Fusion_Conv_Block(num_classes)

        C = self.fusion_1.C
        C_mid = (C + num_classes)//2
        self.prep_out = nn.Sequential(
            nn.Conv2d(C, C_mid, kernel_size=1, bias=False),
            InPlaceABNSync(C_mid),
            nn.Dropout2d(0.1),
            nn.Conv2d(C_mid, num_classes, kernel_size=1, bias=True)
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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)
        x = self.context_encoding(x5)                       #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fusion_1(x)                    #Shape: (8, 1024, 119, 119)
        fusion_result = self.fusion_2(fusion_result)        #Shape: (8, 1024, 119, 119)
        fusion_result = self.fusion_3(fusion_result)        #Shape: (8, 1024, 119, 119)

        fusion_result = self.prep_out(fusion_result)        #Shape: (8, 20, 119, 119)
        
        return [[parsing_result, fusion_result], [edge_result]]

class ResNet_Fusion_Conv_Alt(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet_Fusion_Conv_Alt, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fusion_edges = Fusion_Conv_Block(num_classes, C=768)
        self.fusion_multi = Fusion_Conv_Attention(num_classes, C=1024)

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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)
        x = self.context_encoding(x5)                       #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)

        edge_fea = self.fusion_edges(edge_fea)              #Shape: (8, 768, 119, 119)

        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fusion_multi(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]

class ResNet_Fusion_Conv_Alt_Blocks(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet_Fusion_Conv_Alt_Blocks, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fusion_1 = Fusion_Conv_Block(num_classes)
        self.fusion_2 = Fusion_Conv_Block(num_classes)
        self.fusion_3 = Fusion_Conv_Block(num_classes)
        self.fusion_edges = Fusion_Conv_Block(num_classes, C=768)

        C = self.fusion_1.C
        C_mid = (C + num_classes)//2
        self.prep_out = nn.Sequential(
            nn.Conv2d(C, C_mid, kernel_size=1, bias=False),
            InPlaceABNSync(C_mid),
            nn.Dropout2d(0.1),
            nn.Conv2d(C_mid, num_classes, kernel_size=1, bias=True)
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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)
        x = self.context_encoding(x5)                       #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        edge_fea = self.fusion_edges(edge_fea)              #Shape: (8, 768, 119, 119)

        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fusion_1(x)                    #Shape: (8, 1024, 119, 119)
        fusion_result = self.fusion_2(fusion_result)        #Shape: (8, 1024, 119, 119)
        fusion_result = self.fusion_3(fusion_result)        #Shape: (8, 1024, 119, 119)

        fusion_result = self.prep_out(fusion_result)        #Shape: (8, 20, 119, 119)
        
        return [[parsing_result, fusion_result], [edge_result]]


class ResNet_Fusion_Cross_Attention(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet_Fusion_Cross_Attention, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fusion = Fusion_Cross_Attention(num_classes)

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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)
        x = self.context_encoding(x5)                       #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fusion(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]

class ResNet_MultiScale(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet_MultiScale, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.multiscale_1 = MultiScale_Module(in_channels=2048)
        self.multiscale_2 = MultiScale_Module(in_channels=2048)
        self.multiscale_3 = MultiScale_Module(in_channels=2048)

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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)
        
        x_multiscale = self.multiscale_1(x5)                  #Shape: (8, 2048, 30, 30)   #TODO CHANGED: Added
        x_multiscale = self.multiscale_2(x_multiscale)
        x_multiscale = self.multiscale_3(x_multiscale)
        
        x = self.context_encoding(x_multiscale)             #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fushion(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]

class ResNet_MultiScale_btw3(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet_MultiScale_btw3, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.multiscale_1_1 = MultiScale_Module(in_channels=128)
        self.multiscale_1_2 = MultiScale_Module(in_channels=128)
        self.multiscale_1_3 = MultiScale_Module(in_channels=128)
        self.multiscale_2_1 = MultiScale_Module(in_channels=256)
        self.multiscale_2_2 = MultiScale_Module(in_channels=256)
        self.multiscale_3 = MultiScale_Module(in_channels=512)

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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x = self.multiscale_1_1(x)
        x = self.multiscale_1_2(x)
        x = self.multiscale_1_3(x)

        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x2 = self.multiscale_2_1(x2)
        x2 = self.multiscale_2_2(x2)

        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x3 = self.multiscale_3(x3)

        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)
        x = self.context_encoding(x5)             #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fushion(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]


class ResNet_Mod(nn.Module):            #Model with new modules
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet_Mod, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])                
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fusion = Fusion_Conv_Attention(num_classes)
        self.multiscale = MultiScale_Module(in_channels=2048)

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
        x = self.relu1(self.bn1(self.conv1(x)))             #Shape: (8, 3, 473, 473) -> (8, 64, 237, 237)
        x = self.relu2(self.bn2(self.conv2(x)))             #Shape: (8, 64, 237, 237)
        x = self.relu3(self.bn3(self.conv3(x)))             #Shape: (8, 128, 237, 237)
        x = self.maxpool(x)                                 #Shape: (8, 128, 119, 119)
        x2 = self.layer1(x)                                 #Shape: (8, 256, 119, 119)
        x3 = self.layer2(x2)                                #Shape: (8, 512, 60, 60)
        x4 = self.layer3(x3)                                #Shape: (8, 1024, 30, 30)
        x5 = self.layer4(x4)                                #Shape: (8, 2048, 30, 30)

        x_multiscale = self.multiscale(x5)                  #Shape: (8, 2048, 30, 30)

        x = self.context_encoding(x_multiscale)                       #Shape: (8, 512, 30, 30)
        parsing_result, parsing_fea = self.decoder(x, x2)   #Shapes: (8, 20, 119, 119), (8, 256, 119, 119)
        
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)       #Shapes: (8, 2, 119, 119), (8, 768, 119, 119)
        
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)       #Shape: (8, 1024, 119, 119). Here, 1024 is (3+1)*256.

        fusion_result = self.fusion(x)                     #Shape: (8, 20, 119, 119)

        
        return [[parsing_result, fusion_result], [edge_result]]




def initialize_pretrained_model(model, settings, pretrained='./models/resnet101-imagenet.pth'):
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

    if pretrained is not None:
        saved_state_dict = torch.load(pretrained)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        model.load_state_dict(new_params)


def resnet101(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model

def resnet101_fusion_conv(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet_Fusion_Conv_Attention(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model

def resnet101_fusion_conv_blocks(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet_Fusion_Conv_Blocks(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model

def resnet101_fusion_conv_alt(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet_Fusion_Conv_Alt(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model

def resnet101_fusion_conv_alt_blocks(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet_Fusion_Conv_Alt_Blocks(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model


def resnet101_fusion_cross(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet_Fusion_Cross_Attention(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model

def resnet101_multiscale(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet_MultiScale(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model

def resnet101_multiscale_btw3(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet_MultiScale_btw3(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model


def resnet101_mod(num_classes=20, pretrained='./models/resnet101-imagenet.pth'):
    model = ResNet_Mod(Bottleneck, [3, 4, 23, 3], num_classes)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model
