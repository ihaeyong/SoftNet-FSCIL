import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
import math
import numpy as np

def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()

class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity, smooth):
        with torch.no_grad():

            k_val = percentile(scores.abs(), sparsity*100)
            if smooth:
                zeros = torch.distributions.Uniform(0, 1.0).sample(zeros.size())
            else:
                '''HardNet: zeros'''
                None

            output = torch.where(scores.abs() < k_val,
                                 zeros.to(scores.device),
                                 ones.to(scores.device))

            ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, g):
        #g = F.hardtanh(g * ctx.saved_tensors[0].clone())
        return g, None, None, None, None

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def get_none_masks(model):
    none_masks = {}
    for name, module in model.named_modules():
        if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
            none_masks[name + '.weight'] = None
            none_masks[name + '.bias'] = None

class SignetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, trainable=True, gamma=0.9):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.gamma = gamma
        self.sparsity = 0.1
        self.trainable = trainable
        self.smooth = False

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()
        self.beta = None

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train",
                sparsity=None):
        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode == "train":
            smooth = True if sparsity is True else False
            if weight_mask is None:
                self.weight_mask = GetSubnetFaster.apply(self.w_m,
                                                         self.zeros_weight,
                                                         self.ones_weight,
                                                         self.sparsity,
                                                         self.smooth)
            else:
                self.weight_mask = weight_mask
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = self.sigmoid(self.b_m)
                b_pruned = self.bias_mask * self.bias

        # If inference/valid, use the last compute masks/subnetworks
        if mode == "valid":
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subnetwork
        elif mode == "test":
            if self.beta is not None:
                w_pruned = weight_mask * self.weight * self.beta
            else:
                w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

class SignetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, trainable=True, gamma=0.9):
        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.stride = stride
        self.gamma = gamma
        self.sparsity = 0.1
        self.trainable = trainable
        self.smooth = False

        # Mask Parameters of Weight and Bias
        self.w_m = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()
        self.alpha = None

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train", sparsity=None):
        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode == "train":
            smooth = True if sparsity is True else False
            if weight_mask is None:
                self.weight_mask = GetSubnetFaster.apply(self.w_m,
                                                         self.zeros_weight,
                                                         self.ones_weight,
                                                         self.sparsity,
                                                         self.smooth)
            else:
                self.weight_mask = weight_mask

            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = self.b_m
                b_pruned = self.bias_mask * self.bias

        # If inference/valid, use the last compute masks/subnetworks
        elif mode == "valid":
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference/test, no need to compute the subnetwork
        elif mode == "test":
            if weight_mask is None:
                pass

            if self.alpha is not None:
                w_pruned = weight_mask * self.weight * self.alpha
            else:
                w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported!")

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)
