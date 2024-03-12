import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, adopt_classifier=True,
                           flatten=True):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.flatten = flatten
        self.adopt_classifier = adopt_classifier

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if self.adopt_classifier:
            self.fc = nn.Linear(64, num_classes)
            nn.init.normal_(self.fc.weight, std=0.001)
            nn.init.constant_(self.fc.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        output = self.avg_pool(x)

        if self.flatten:
            output = output.view(output.size(0), -1)
            if self.adopt_classifier:
                output = self.fc(output)\

        return output

    def forward_without_cf(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        output = self.avg_pool(x)

        output = output.view(output.size(0), -1)

        return output

    def forward_o_embeddings(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        output = self.avg_pool(x)

        if self.flatten:
            output = output.view(output.size(0), -1)
            if self.adopt_classifier:
                y = self.fc(output)

        return output, y

def resnet20(**kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

class Resnet20_cifar_small(nn.Module):
    def __init__(self, num_classes=100, adopt_classifier=True, flatten=True):
        super(Resnet20_cifar_small, self).__init__()
        self.func = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, adopt_classifier=adopt_classifier,
                           flatten=flatten)

    def forward(self, x):
        return self.func(x)

    def forward_without_cf(self, x):
        return self.func.forward_without_cf(x)

    def forward_o_embeddings(self, x):
        return self.func.forward_o_embeddings(x)
