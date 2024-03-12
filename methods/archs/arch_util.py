import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init

def init_modules(module_list, bias_fill=0):
    if not isinstance(module_list, list):
        module_list = [module_list]

    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class AvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(AvgPool2d, self).__init__()
        self.m = torch.nn.AdaptiveAvgPool2d(output_size)
    def forward(self, x):
        x = self.m(x)
        x = x.view(x.size(0), -1)
        return x