#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：step_utils.py
@Author  ：wkml4996
@Date    ：2021/10/6 22:24 
"""
from mindspore import nn
from mindspore import ops
from mindspore import ParameterTuple


class TrainWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(TrainWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data1, data2, label):
        out = self._backbone(data1, data2)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        # 使用tuple包装weight
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        # 定义梯度函数
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, x_1, x_2, label):
        """构建训练过程"""
        loss = self.network(x_1, x_2, label)
        grads = self.grad(self.network, self.weights)(x_1, x_2, label)
        return loss, self.optimizer(grads)


class GradNetWithWrtParams(nn.Cell):
    def __init__(self, net):
        super(GradNetWithWrtParams, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)
