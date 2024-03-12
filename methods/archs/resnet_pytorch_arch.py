import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from copy import deepcopy
import torchvision
from torch.nn.parameter import Parameter

class ResNet_pytorch(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=False, cut_at_pooling=False,
                 Embed_dim=0, norm=True, num_classes=0, adopt_classifier=False,
                 addition_net=True, flatten=True):
        super(ResNet_pytorch, self).__init__()
        self.layers = []
        self.Embed_dim = Embed_dim  # lu adds
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.adopt_classifier = adopt_classifier
        self.flatten = flatten


        # Construct base (pretrained) resnet
        if depth not in ResNet_pytorch.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet_pytorch.__factory[depth](pretrained=pretrained)
        #if not self.cut_at_pooling:
        self.num_features = Embed_dim
        self.norm = norm
        self.addition_net = addition_net
        self.num_classes = num_classes

        out_planes = self.base.fc.in_features

        # Append new layers
        if self.addition_net:
            self.base.fc = nn.Linear(out_planes, self.num_features)
            init.kaiming_normal_(self.base.fc.weight, mode='fan_out')
            init.constant_(self.base.fc.bias, 0)
            # Change the num_features to CNN output channels
        self.num_features = out_planes
        if self.num_classes > 0 and self.adopt_classifier:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()


    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])

        if self.flatten:
            x = x.view(x.size(0), -1)
            if self.addition_net:
                x = self.base.fc(x)

            if self.norm:
                x = F.normalize(x)

            if self.num_classes > 0 and self.adopt_classifier:
                x = F.relu(x)
                x = self.classifier(x)
        return x

    def forward_without_cf(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])

        x = x.view(x.size(0), -1)

        return x

    def forward_o_embeddings(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])

        x = x.view(x.size(0), -1)
        embedding = x

        if self.addition_net:
            x = self.base.fc(x)

        if self.norm:
            x = F.normalize(x)

        if self.num_classes > 0 and self.adopt_classifier:
            x = F.relu(x)
            x = self.classifier(x)
        return embedding, x

    def inference(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.base.fc(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)

        if self.num_classes > 0:
            x = F.relu(x)

        return x

    def forward_without_norm(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.base.fc(x)
        if self.num_classes > 0:
            x = F.relu(x)
            x = self.classifier(x)

        return x

    def extract_feat(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def unfoldLayer(self, module):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """

        # get all layers of the model
        layer_list = list(module.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0:
                self.layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, torch.nn.Module):
                self.unfoldLayer(module)

class Resnet18_pretrain(nn.Module):
    def __init__(self):
        super(Resnet18_pretrain, self).__init__()
        self.func = ResNet_pytorch(18, pretrained=True, Embed_dim=512)

    def forward(self, x):
        return self.func(x)

class Resnet18_softmax(nn.Module):
    def __init__(self, Embed_dim=512, cut_at_pooling=False, pretrained=False, norm=True, num_classes=50,
                 adopt_classifier=True, addition_net=True, flatten=True):
        super(Resnet18_softmax, self).__init__()
        self.func = ResNet_pytorch(18, pretrained=pretrained, cut_at_pooling=cut_at_pooling, norm=norm,
                                   Embed_dim=Embed_dim, num_classes=num_classes,
                                   adopt_classifier=adopt_classifier, addition_net=addition_net,
                                   flatten=flatten)
    def forward(self, x):
        return self.func(x)

    def forward_without_cf(self, x):
        return self.func.forward_without_cf(x)

    def forward_o_embeddings(self, x):
        return self.func.forward_o_embeddings(x)
