import argparse
import os
import time
from logging import getLogger
import torch.nn.functional as F
import torch
import torch.nn as nn


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, num_labels, arch="vid_base_arch",use_lincls_use_bn=True, use_lincls_l2_norm=False,lincls_drop=0.0):
        super(RegLog, self).__init__()
        self.use_lincls_l2_norm = use_lincls_l2_norm
        self.lincls_drop = lincls_drop
        self.use_lincls_use_bn = use_lincls_use_bn
        if arch == "r2plus1d_18_old":
            s = 512
        elif arch in  ["r2plus1d_18_custom",'s3d_new']:
            s = 1024*2
        if self.use_lincls_use_bn:
            # self.bn = nn.BatchNorm2d(s)
            self.bn = nn.BatchNorm1d(s,momentum=0.1)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()

        if self.lincls_drop>0:
            self.drop_lyr = nn.Dropout(p=self.lincls_drop)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        if self.use_lincls_l2_norm:
            x = F.normalize(x, p=2, dim=1)
        # optional BN
        if self.use_lincls_use_bn:
            # print('BN running')
            x = self.bn(x)
        if self.lincls_drop > 0:
            x=self.drop_lyr(x)
        # print('x shape:',x.shape)
        # flatten
        # x = x.view(x, -1)

        # linear layer
        return self.linear(x)

