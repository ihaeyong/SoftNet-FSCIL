# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from .arch_util import init_modules
import torchvision

class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if not hasattr(self.weight, 'fast'):
                self.weight.fast = None
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if not hasattr(self.weight, 'fast'):
            self.weight.fast = None
        if not hasattr(self.bias, 'fast'):
            self.bias.fast = None
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=0.1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            #out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
            out = super(BatchNorm2d_fw, self).forward(x)
        return out


# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = True  # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        init_modules(self.parametrized_layers)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


class ResNet_meta(nn.Module):
    __factory = {

        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    maml = True  # Default

    def __init__(self, depth, block, list_of_num_layers, list_of_out_dims, flatten=True, Embed_dim=512,
                 pretrained=True, norm=False, num_classes=100, adopt_classifier=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage

        self.flatten = flatten
        self.Embed_dim = Embed_dim
        self.norm = norm
        self.num_classes = num_classes
        self.adopt_classifier = adopt_classifier
        self.pretrained = pretrained

        super(ResNet_meta, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_modules(conv1)
        init_modules(bn1)

        base = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                base.append(B)
                indim = list_of_out_dims[i]

        self.base = nn.Sequential(*base)

        if self.num_classes > 0 and self.adopt_classifier:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init_modules(self.classifier)

        if self.pretrained:
            self.load_pretrained_model(depth)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        if self.flatten:
            x = x.view(x.size(0), -1)

            if self.norm:
                x = F.normalize(x)

            if self.num_classes > 0 and self.adopt_classifier:
                x = F.relu(x)
                x = self.classifier(x)
        return x

    def load_pretrained_model(self, depth):
        if depth not in ResNet_meta.__factory:
            raise KeyError("Unsupported depth:", depth)
        pt_model = ResNet_meta.__factory[depth](pretrained=self.pretrained)

        parameter_list = [parameter for name, parameter in pt_model.named_parameters() if 'fc' not in name]
        for id, (name, parameter) in enumerate(self.named_parameters()):
            parameter.data = parameter_list[id]


class Resnet18_meta(nn.Module):
    def __init__(self, flatten=True, Embed_dim=512, pretrained=True, norm=False, num_classes=100, adopt_classifier=False):
        super(Resnet18_meta, self).__init__()
        self.func = ResNet_meta(18, SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten=flatten,
                                  Embed_dim=Embed_dim, pretrained=pretrained,
                                  norm=norm, num_classes=num_classes, adopt_classifier=adopt_classifier)

    def forward(self, x):
        return self.func(x)


