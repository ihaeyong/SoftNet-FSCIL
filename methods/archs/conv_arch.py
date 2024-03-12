from torch import nn as nn
from torch.nn import functional as F
from methods.archs import arch_util as arch_util
from torch.nn import init

# Simple Conv Block
class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
        self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        arch_util.init_modules(self.parametrized_layers)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True, num_classes=0, adopt_classifier=False):
        super(ConvNet,self).__init__()
        self.num_classes = num_classes
        self.adopt_classifier = adopt_classifier
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i < 4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(nn.AdaptiveAvgPool2d(1))
            trunk.append(arch_util.Flatten())

        if self.num_classes > 0:
            self.classifier = nn.Linear(64, self.num_classes)
            init.normal(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self,x):

        x = self.trunk(x)
        if self.num_classes > 0 and self.adopt_classifier:
            x = self.classifier(x)

        return x

class Conv6(nn.Module):
    def __init__(self, flatten = True, num_classes=0, adopt_classifier=False):
        super(Conv6, self).__init__()
        self.fun = ConvNet(depth=6, flatten=flatten, num_classes=num_classes, adopt_classifier=adopt_classifier)

    def forward(self, x):
        out = self.fun(x)
        return out

