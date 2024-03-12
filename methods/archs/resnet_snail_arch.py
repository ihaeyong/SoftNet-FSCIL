import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channels * BasicBlock.expansion)

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)

        out = self.relu(out)

        return out

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, block_size, in_channels, num_classes=100, adopt_classifier=True, flatten=True):
        super(ResNet, self).__init__()

        self.adopt_classifier = adopt_classifier
        self.flatten = flatten

        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(3, block_size[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(block_size[0])
        self.relu = nn.ReLU(inplace=True)

        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, block_size[0], num_block[0], 1)
        self.conv3_x = self._make_layer(block, block_size[1], num_block[1], 2)
        self.conv4_x = self._make_layer(block, block_size[2], num_block[2], 2)
        self.conv5_x = self._make_layer(block, block_size[3], num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if self.adopt_classifier:
            self.fc = nn.Linear(block_size[3] * block.expansion, num_classes)
            nn.init.normal_(self.fc.weight, std=0.001)
            nn.init.constant_(self.fc.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)

        if self.flatten:
            output = output.view(output.size(0), -1)
            if self.adopt_classifier:
                output = self.fc(output)

        return output

    def forward_without_cf(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)

        output = output.view(output.size(0), -1)
        return output

    def forward_o_embeddings(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)

        if self.flatten:
            output = output.view(output.size(0), -1)
            if self.adopt_classifier:
                y = self.fc(output)
            else:
                raise ValueError('The adopt_classifier should be True if call function forward_o_embeddings!')

        return output, y

# class Resnet18_cifar(nn.Module):
#     def __init__(self, num_classes=100, adopt_classifier=True, flatten=True):
#         super(Resnet18_cifar, self).__init__()
#         self.func = ResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], in_channels=64, num_classes=num_classes, adopt_classifier=adopt_classifier,
#                            flatten=flatten)
#
#     def forward(self, x):
#         return self.func(x)
#
#     def forward_without_cf(self, x):
#         return self.func.forward_without_cf(x)
#
#     def forward_o_embeddings(self, x):
#         return self.func.forward_o_embeddings(x)

class MiniImagenetNet(nn.Module):
    def __init__(self, num_classes=100, adopt_classifier=True, flatten=True):
        super(MiniImagenetNet, self).__init__()
        self.func = ResNet(BasicBlock, [1, 1, 1, 1], [64, 96, 128, 256], in_channels=64, num_classes=num_classes, adopt_classifier=adopt_classifier,
                           flatten=flatten)

    def forward(self, x):
        return self.func(x)

    def forward_without_cf(self, x):
        return self.func.forward_without_cf(x)

    def forward_o_embeddings(self, x):
        return self.func.forward_o_embeddings(x)

class MiniImagenetNet_v2(nn.Module):
    def __init__(self, num_classes=100, adopt_classifier=True, flatten=True):
        super(MiniImagenetNet_v2, self).__init__()
        self.func = ResNet(BasicBlock, [1, 1, 1, 1], [64, 96, 128, 256], in_channels=64, num_classes=num_classes, adopt_classifier=adopt_classifier,
                           flatten=flatten)

    def forward(self, x):
        return self.func(x)

    def forward_without_cf(self, x):
        return self.func.forward_without_cf(x)

    def forward_o_embeddings(self, x):
        return self.func.forward_o_embeddings(x)
