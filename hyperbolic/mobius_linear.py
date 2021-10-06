# -*- coding: utf-8 -*-
# Time : 2021/9/25 11:03
# Author : Horace
# File : mobius_linear.py
# blog : horacehht.github.io
import math
import mindspore as ms
from mindspore import nn, Parameter, Tensor
import mindspore.numpy as np
import numpy


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, gain: float = 1.):
    """HyperText用"""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    boundary = gain * math.sqrt(6.0 / float(fan_in + fan_out))
    data = Tensor(numpy.random.uniform(-boundary, boundary, tensor.shape),dtype=ms.float32)
    return data


def xavier_normal(tensor, gain: float = 1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain*math.sqrt(2.0 / float(fan_in + fan_out))
    data = numpy.random.normal(0, std, tensor.shape)
    return data


class MobiusLinear(nn.Cell):
    """
        Mobius linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, use_bias=True):
        """

        :type c: object
        """
        super(MobiusLinear, self).__init__()
        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.manifold = manifold
        self.bias = Parameter(np.zeros(out_features,dtype=ms.float32))

        self.weight = Parameter(Tensor(numpy.random.random(size=(out_features, in_features)),dtype=ms.float32))
        self.reset_parameters()

    # 不知道没有torch.no_grad是否会有影响？
    def reset_parameters(self):
        self.weight = xavier_uniform(self.weight, gain=math.sqrt(2))
        self.bias.set_data(np.zeros(self.bias.shape))  # 重新初始化bias为全0向量

    def construct(self, x):
        # 这里就涉及到Poincare的很多函数用法了，不是我这个模块的东西了
        mv = self.manifold.mobius_matvec(self.weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1))
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features_size={}, out_features_size={}, curvalture={}'.format(
            self.in_features, self.out_features, self.c
        )