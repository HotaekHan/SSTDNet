import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from inception import load_inception

class SSTDnet(nn.Module):
    num_anchors = 9
    num_classes = 10
    num_head_layers = 4

    def __init__(self, num_classes, pretrained):
        super(SSTDnet, self).__init__()
        self.num_classes = num_classes

        self.basenet = load_inception(pretrained)

        self.conv1_mask = nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=False)
        self.conv1_bn_mask = nn.BatchNorm2d(512)
        self.deconv_mask = nn.ConvTranspose2d(512, 2, kernel_size=16, padding=4, stride=8, groups=2)
        self.conv2_mask = nn.Conv2d(2, 2, kernel_size=3, padding=2, stride=1, dilation=2, bias=False)
        self.conv2_bn_mask = nn.BatchNorm2d(2)
        self.conv3_mask = nn.Conv2d(2, 2, kernel_size=3, padding=2, stride=1, dilation=2)
        self.softmax_mask = nn.Softmax2d()

        self.deconv_fc7 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False, groups=64)
        self.deconv_conv6 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False, groups=64)
        self.deconv_conv7 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False, groups=64)

        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def _make_head(self, output_dim):
        layers = []

        layers.append(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
        for _ in range(self.num_head_layers - 1):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, output_dim, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        conv3, conv4, fc7, conv6, conv7, conv8, conv9, conv10 = self.basenet(x)

        down_conv3 = F.max_pool2d(conv3, kernel_size=3, stride=2, padding=1)
        up_fc7 = self.deconv_fc7(fc7)

        down_conv4 = F.max_pool2d(conv4, kernel_size=3, stride=2, padding=1)
        up_conv6 = self.deconv_conv6(conv6)

        down_fc7 = F.max_pool2d(fc7, kernel_size=3, stride=2, padding=1)
        up_conv7 = self.deconv_conv7(conv7)

        aif_1 = down_conv3 + conv4 + up_fc7
        aif_2 = down_conv4 + fc7 + up_conv6
        aif_3 = down_fc7 + conv6 + up_conv7

        mask = down_conv3 + conv4 + up_fc7
        mask = F.relu(self.conv1_bn_mask(self.conv1_mask(mask)))
        mask = self.deconv_mask(mask)
        mask = F.relu(self.conv2_bn_mask(self.conv2_mask(mask)))
        mask = self.conv3_mask(mask)

        attention = self.softmax_mask(mask)
        attention = attention[:,1:2,:,:]
        attention = F.avg_pool2d(attention, kernel_size=4, stride=4)

        attention = F.avg_pool2d(attention, kernel_size=2, stride=2)
        masked_aif_1 = attention * aif_1
        attention = F.avg_pool2d(attention, kernel_size=2, stride=2)
        masked_aif_2 = attention * aif_2
        attention = F.avg_pool2d(attention, kernel_size=2, stride=2)
        masked_aif_3 = attention * aif_3

        # fms = [masked_aif_1, masked_aif_2, masked_aif_3, conv7, conv8, conv9, conv10]
        fms = [masked_aif_1, masked_aif_2, masked_aif_3, conv7, conv8, conv9]

        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0), -1, 4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0), -1, self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1), mask


def load_sstdnet(num_classes, using_pretrained=True):
    return SSTDnet(num_classes=num_classes, pretrained=using_pretrained)

def test():
    net = SSTDnet(num_classes=10, pretrained=True)

    loc_preds, cls_preds, attention = net(Variable(torch.randn(1,3,512,512)))
    print(loc_preds.size())
    print(cls_preds.size())
    print(attention.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    att_grads = Variable(torch.randn(attention.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)
    attention.backward(att_grads)
# test()

