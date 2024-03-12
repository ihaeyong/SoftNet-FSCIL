import torch
from torch import nn
from torch.nn import functional as F
from .arch_util import init_modules, Flatten

class TFUnit(nn.Module):
    def __init__(self, Embed_dim=512, kernel_size=1):
        super(TFUnit, self).__init__()
        self.flatten = Flatten()
        self.Embed_dim = Embed_dim
        self.kernel_size = kernel_size
        if self.kernel_size == 1:
            conv_block = nn.Conv2d(self.Embed_dim, self.Embed_dim, 1)
        elif self.kernel_size == 3:
            conv_block = nn.Conv2d(self.Embed_dim, self.Embed_dim, 3, padding=1)
        else:
            raise ValueError(f'the kernel size of {self.__class__.__name__} can not be {self.kernel_size}')
        bn = nn.BatchNorm2d(self.Embed_dim)
        relu = nn.ReLU(inplace=True)
        parametrized_layers = [conv_block, bn, relu]
        init_modules(parametrized_layers)
        self.trunk = nn.Sequential(*parametrized_layers)

    def forward(self, x):
        x = self.trunk(x)
        return x

class TFNet(nn.Module):
    __factory = {
        'TFUnit': TFUnit,
    }

    def __init__(self, num_tf_unit, tf_unit_opt):
        super(TFNet, self).__init__()
        self.num_tf_unit = 0
        self.tf_unit_opt = tf_unit_opt
        self.tf_unit_type = tf_unit_opt.pop('type')

        self.unit = TFNet.__factory[self.tf_unit_type]

        # ModuleList support dynamic add or delete layers
        self.tf_unit_list = nn.ModuleList()
        self.__init_net(num_tf_unit=num_tf_unit)


    def forward(self,x):
        """
        x (torch.Tensor): the input should be a 1D vector
        """
        # reshape the vector x
        for i, tf in enumerate(self.tf_unit_list):
            x = tf(x)
        return x

    def __init_net(self, num_tf_unit):
        for i in range(num_tf_unit):
            tf_unit = self.unit(**self.tf_unit_opt)
            self.add_tf_unit(tf_unit=tf_unit)

    def add_tf_unit(self, tf_unit):
        self.tf_unit_list.append(tf_unit)
        self.num_tf_unit += 1