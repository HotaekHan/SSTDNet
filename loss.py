'''
Reference :
   https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):
    num_classes = 10

    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.using_gpu = torch.cuda.is_available()

    def focal_loss2d(self, x, y):
        alpha = 0.25
        gamma = 2

        y = y.view(-1)
        x = x.permute(0,2,3,1).contiguous().view(-1, 2)

        if self.using_gpu is True:
            t = one_hot_embedding(y.data.cpu(), 2)
        else:
            t = one_hot_embedding(y.data, 2)
        # t = t[:,1:]  # exclude background

        if self.using_gpu is True:
            t = Variable(t).cuda()
        else:
            t = Variable(t)

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)

        return F.binary_cross_entropy_with_logits(x, t, w)

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        if self.using_gpu is True:
            t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        else:
            t = one_hot_embedding(y.data, 1+self.num_classes)
        t = t[:,1:]  # exclude background

        if self.using_gpu is True:
            t = Variable(t).cuda()
        else:
            t = Variable(t)

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)

        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.

        This implementation is numerically unstable because of log function.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        t = Variable(t).cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = ((2*xt+1).sigmoid()+0.0001).data.clamp_(min=0, max=1)

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        # mask_loss = self.ce_loss(mask_preds, mask_targets)
        mask_loss = self.focal_loss2d(mask_preds, mask_targets)


        print('loc_loss: %.3f | cls_loss: %.3f | mask_loss: %.3f' %
              (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos, mask_loss.data[0]), end=' | ')
        loss = ((loc_loss+cls_loss)/num_pos) + (mask_loss)
        return loss, num_pos
