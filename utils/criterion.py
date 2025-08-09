#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   criterion.py
@Time    :   8/30/19 8:59 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .lovasz_softmax import LovaszSoftmax
from .kl_loss import KLDivergenceLoss
from .consistency_loss import ConsistencyLoss, ConsistencyLoss_Soft

NUM_CLASSES = 20


class CriterionAll(nn.Module):
    def __init__(self, use_class_weight=False, ignore_index=255, lambda_1=1, lambda_2=1, lambda_3=1,
                 cons_loss_type="hard",
                 num_classes=20):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.use_class_weight = use_class_weight
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)
        self.kldiv = KLDivergenceLoss(ignore_index=ignore_index)
        self.reg = ConsistencyLoss(ignore_index=ignore_index) if cons_loss_type == "hard" else ConsistencyLoss_Soft(ignore_index=ignore_index)
        self.lamda_1 = lambda_1
        self.lamda_2 = lambda_2
        self.lamda_3 = lambda_3
        self.num_classes = num_classes

    def parsing_loss(self, preds, target, cycle_n=None):
        """
        Loss function definition.

        Args:
            preds: [[parsing result1, parsing result2],[edge result]]
            target: [parsing label, egde label]
            soft_preds: [[parsing result1, parsing result2],[edge result]]
        Returns:
            Calculated Loss.
        """
        h, w = target[0].size(1), target[0].size(2)                 #Size of GT segmentation

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)      #Number of edge pixels in GT edges
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)      #Number of non-edge pixels in GT edges

        weight_pos = neg_num / (pos_num + neg_num)                  #Ratio of non-edge pixels in GT edges
        weight_neg = pos_num / (pos_num + neg_num)                  #Ratio of edge pixels in GT edges
        weights = torch.tensor([weight_neg, weight_pos])  # edge loss weight

        loss = 0

        # loss for segmentation
        #We have two loss contributions to the parsing loss. 
        # One is comparison with target, and the other with soft_preds (updated labels)
        preds_parsing = preds[0]
        for pred_parsing in preds_parsing:                                      #Goes over both the parsing and fusion results
            scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)     #Shape: (B, C, H, W)

            loss += 0.5 * self.lamda_1 * self.lovasz(scale_pred, target[0])     #miou loss
            if target[2] is None:
                loss += 0.5 * self.lamda_1 * self.criterion(scale_pred, target[0])      #Class loss. KL-Div btw pred and GT label
            else:
                soft_scale_pred = F.interpolate(input=target[2], size=(h, w),
                                                mode='bilinear', align_corners=True)
                soft_scale_pred = moving_average(soft_scale_pred, to_one_hot(target[0], num_cls=self.num_classes),
                                                 1.0 / (cycle_n + 1.0))
                loss += 0.5 * self.lamda_1 * self.kldiv(scale_pred, soft_scale_pred, target[0])     #KL-Div btw preds and soft_preds

        loss_seg = loss.detach().clone()

        # loss for edge
        preds_edge = preds[1]
        for pred_edge in preds_edge:
            scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            if target[3] is None:
                loss += self.lamda_2 * F.cross_entropy(scale_pred, target[1],
                                                       weights.cuda(), ignore_index=self.ignore_index)
            else:
                soft_scale_edge = F.interpolate(input=target[3], size=(h, w),
                                                mode='bilinear', align_corners=True)
                soft_scale_edge = moving_average(soft_scale_edge, to_one_hot(target[1], num_cls=2),
                                                 1.0 / (cycle_n + 1.0))
                loss += self.lamda_2 * self.kldiv(scale_pred, soft_scale_edge, target[0])       #KL-Div btw edge_preds and soft_edge_preds

        loss_edge = loss.detach().clone() - loss_seg

        # consistency regularization
        preds_parsing = preds[0]
        preds_edge = preds[1]
        for pred_parsing in preds_parsing:                                          #Iterates through [parsing_result, fusion_result]
            scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            scale_edge = F.interpolate(input=preds_edge[0], size=(h, w),
                                       mode='bilinear', align_corners=True)
            
            cons_term = self.reg(scale_pred, scale_edge, target[0])
            loss += self.lamda_3 * cons_term
            
            #loss += self.lamda_3 * self.reg(scale_pred, scale_edge, target[0])

        loss_consistency = loss.detach().clone() - loss_edge - loss_seg

        return loss, loss_seg, loss_edge, loss_consistency

    def forward(self, preds, target, cycle_n=None):
        # loss = self.parsing_loss(preds, target, cycle_n)
        # return loss

        loss, loss_seg, loss_edge, loss_cons = self.parsing_loss(preds, target, cycle_n)
        return loss, loss_seg, loss_edge, loss_cons

    def _generate_weights(self, masks, num_classes):
        """
        masks: torch.Tensor with shape [B, H, W]
        """
        masks_label = masks.data.cpu().numpy().astype(np.int64)
        pixel_nums = []
        tot_pixels = 0
        for i in range(num_classes):
            pixel_num_of_cls_i = np.sum(masks_label == i).astype(np.float)
            pixel_nums.append(pixel_num_of_cls_i)
            tot_pixels += pixel_num_of_cls_i
        weights = []
        for i in range(num_classes):
            weights.append(
                (tot_pixels - pixel_nums[i]) / tot_pixels / (num_classes - 1)
            )
        weights = np.array(weights, dtype=np.float)
        # weights = torch.from_numpy(weights).float().to(masks.device)
        return weights


def moving_average(target1, target2, alpha=1.0):
    target = 0
    target += (1.0 - alpha) * target1
    target += target2 * alpha
    return target


def to_one_hot(tensor, num_cls, dim=1, ignore_index=255):
    b, h, w = tensor.shape
    tensor[tensor == ignore_index] = 0
    onehot_tensor = torch.zeros(b, num_cls, h, w).cuda()
    onehot_tensor.scatter_(dim, tensor.unsqueeze(dim), 1)
    return onehot_tensor
