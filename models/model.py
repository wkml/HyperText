#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：model.py
@Author  ：wkml4996
@Date    ：2021/9/28 21:18 
"""
from mindspore.common.initializer import initializer, XavierUniform
from poincare import *
from mobius_linear import *
from mindspore import ops
from mindspore.ops import Concat
import numpy as np

def xavier_normal(tensor, gain: float = 1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain*math.sqrt(2.0 / float(fan_in + fan_out))
    data = numpy.random.normal(0, std, tensor.shape)
    return data


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

class Model(nn.Cell):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.c_seed = 1.0
        self.manifold = PoincareBall()
        self.op = ops.Concat(1)
        if config.embedding_pretrained is not None:
            pass
        else:
            emb = Tensor(xavier_normal(Tensor(np.random.random((config.n_vocab, config.embed)))),dtype=ms.float32)
            # emb = Tensor(xavier_normal(Tensor(np.random.random((500, config.embed)))))
            emb[0] = emb[0].fill(0)
            self.emb = emb
            # self.embedding = ManifoldParameter(emb, requires_grad=True, manifold=self.manifold, c=self.c_seed)

        emb_wordngram = Tensor(xavier_normal(Tensor(np.random.random((config.bucket, config.embed)))),dtype=ms.float32)
        # emb_wordngram = Tensor(xavier_normal(Tensor(np.random.random((500, config.embed)))))
        emb_wordngram[0] = emb_wordngram[0].fill(0)
        self.emb_wordngram = emb_wordngram
        # self.embedding_wordngram = ManifoldParameter(emb_wordngram, requires_grad=True, manifold=self.manifold,
        #                                              c=self.c_seed)
        # 这里的drop传入参数意义与pytorch相反，在原文中传入default为0.0，也就是说在此处需要default为1.0
        self.dropout = nn.Dropout(config.dropout)
        self.hyperLinear = MobiusLinear(self.manifold, config.embed,
                                        config.num_classes, c=self.c_seed)


    def construct(self, x):
        out_word = self.emb[x['id']]
        out_wordngram = self.emb_wordngram[x['ngram']]
        out = self.op((out_word, out_wordngram))
        out = self.manifold.einstein_midpoint(out, c=self.c_seed)
        out = self.hyperLinear(out)
        out = self.manifold.logmap0(out, self.c_seed)

        return out
