import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.autograd import Variable

import torchvision.models as models

class BottleneckA(nn.Module):
    def __init__(self, input_dims):
        super(BottleneckA, self).__init__()

        self.conv1 = nn.Conv2d(input_dims, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2_dilation = nn.Conv2d(input_dims, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(input_dims, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_1 = nn.Conv2d(input_dims, 128, kernel_size=(1,5), stride=1, padding=(0, 4), dilation=2, bias=False)
        self.conv4_1_bn = nn.BatchNorm2d(128)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=(5,1), stride=1, padding=(4, 0), dilation=2, bias=False)
        self.conv4_2_bn = nn.BatchNorm2d(128)

    def forward(self, x):
        c1_out = F.relu(self.conv1_bn(self.conv1(x)))
        c2_out = F.relu(self.conv2_bn(self.conv2_dilation(x)))
        c3_out = F.relu(self.conv3_bn(self.conv3(F.max_pool2d(x, kernel_size=3, stride=1, padding=1))))
        c4_out = F.relu(self.conv4_1_bn(self.conv4_1(x)))
        c4_out = F.relu(self.conv4_2_bn(self.conv4_2(c4_out)))

        output = [c1_out, c2_out, c3_out, c4_out]

        return torch.cat(output, dim=1)

class BottleneckB(nn.Module):
    def __init__(self, input_dims):
        super(BottleneckB, self).__init__()

        self.conv1 = nn.Conv2d(input_dims, 128, kernel_size=1, stride=1, padding=0)
        self.conv2_dilation = nn.Conv2d(input_dims, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(input_dims, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_1 = nn.Conv2d(input_dims, 128, kernel_size=(1,5), stride=1, padding=(0, 2))
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=(5,1), stride=1, padding=(2, 0))

    def forward(self, x):
        c1_out = F.relu(self.conv1(x))
        c2_out = F.relu(self.conv2_dilation(x))
        c3_out = F.relu(self.conv3(F.max_pool2d(x, kernel_size=3, stride=1, padding=1)))
        c4_out = F.relu(self.conv4_1(x))
        c4_out = F.relu(self.conv4_2(c4_out))

        output = [c1_out, c2_out, c3_out, c4_out]

        return torch.cat(output, dim=1)

class Inception(nn.Module):

    def __init__(self, blockA, BlockB):
        super(Inception, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3_inception = blockA(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3_inception = blockA(512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.fc7_inception = blockA(1024)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=3, stride=2, padding=1)
        self.conv6_1_inception = BlockB(256)
        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv7_1_inception = BlockB(256)
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv8_1_inception = BlockB(256)
        self.conv9_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv9_1_inception = BlockB(256)
        self.conv10_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv10_1_inception = BlockB(256)

    def forward(self, x):
        out = F.relu(self.conv1_1(x))
        out = F.relu(self.conv1_2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        conv3_inception = self.conv3_3_inception(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_inception = self.conv4_3_inception(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
        out = F.relu(self.fc6(out))
        out = F.relu(self.fc7(out))
        fc7_inception = self.fc7_inception(out)
        out = F.relu(self.conv6_1(out))
        conv6_inception = self.conv6_1_inception(out)
        out = F.relu(self.conv7_1(conv6_inception))
        conv7_inception = self.conv7_1_inception(out)
        out = F.relu(self.conv8_1(conv7_inception))
        conv8_inception = self.conv8_1_inception(out)
        out = F.relu(self.conv9_1(conv8_inception))
        conv9_inception = self.conv9_1_inception(out)
        out = F.relu(self.conv10_1(conv9_inception))
        conv10_inception = self.conv10_1_inception(out)

        return conv3_inception, conv4_inception, fc7_inception, conv6_inception, conv7_inception, conv8_inception, conv9_inception, conv10_inception

def load_inception(using_pretrained):
    net = Inception(blockA=BottleneckA, BlockB=BottleneckB)

    if using_pretrained is True:
        pre_trained_vgg16 = models.vgg16(pretrained=True)
        pre_trained_feature = pre_trained_vgg16.features

        net.conv1_1.weight = pre_trained_feature[0].weight
        net.conv1_1.bias = pre_trained_feature[0].bias
        net.conv1_2.weight = pre_trained_feature[2].weight
        net.conv1_2.bias = pre_trained_feature[2].bias

        net.conv2_1.weight = pre_trained_feature[5].weight
        net.conv2_1.bias = pre_trained_feature[5].bias
        net.conv2_2.weight = pre_trained_feature[7].weight
        net.conv2_2.bias = pre_trained_feature[7].bias

        net.conv3_1.weight = pre_trained_feature[10].weight
        net.conv3_1.bias = pre_trained_feature[10].bias
        net.conv3_2.weight = pre_trained_feature[12].weight
        net.conv3_2.bias = pre_trained_feature[12].bias
        net.conv3_3.weight = pre_trained_feature[14].weight
        net.conv3_3.bias = pre_trained_feature[14].bias

        net.conv4_1.weight = pre_trained_feature[17].weight
        net.conv4_1.bias = pre_trained_feature[17].bias
        net.conv4_2.weight = pre_trained_feature[19].weight
        net.conv4_2.bias = pre_trained_feature[19].bias
        net.conv4_3.weight = pre_trained_feature[21].weight
        net.conv4_3.bias = pre_trained_feature[21].bias

        net.conv5_1.weight = pre_trained_feature[24].weight
        net.conv5_1.bias = pre_trained_feature[24].bias
        net.conv5_2.weight = pre_trained_feature[26].weight
        net.conv5_2.bias = pre_trained_feature[26].bias
        net.conv5_3.weight = pre_trained_feature[28].weight
        net.conv5_3.bias = pre_trained_feature[28].bias

    return net


def test():
    net = load_inception(using_pretrained=True)

    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param

    print(net)
    print("num. of parameters : " + str(num_parameters))

    fms = net(Variable(torch.randn(1,3,512,512)))
    for fm in fms:
        print(fm.size())

# test()