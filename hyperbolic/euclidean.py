# -*- coding: utf-8 -*- 
# Time : 2021/9/25 11:01 
# Author : Horace 
# File : euclidean.py
# blog : horacehht.github.io
import mindspore.numpy as np


class Euclidean(object):
    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.shape[-1]
        q = p.view(-1, dim)
        # 归一化，每行加起来的二范数不超过1
        _sum = np.sqrt((q ** 2).sum(axis=1))
        for i in range(len(q)):
            q[i] = q[i] / _sum[i]
        return q

    def sqdist(self, p1, p2, c):
        """计算欧几里得距离"""
        return ((p1 - p2) ** 2).sum(axis=-1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        """
        general add in Euclidean Space
        :param x:
        :param y:
        :param c:curvalture
        :param dim:
        :return:
        """
        return x + y

    def mobius_matvec(self, m, x, c):
        """
        general matrix multiplication
        :param m:
        :param x:
        :param c:curvalture
        :return:
        """
        mx = np.dot(x, m.transpose())  # 暂时还没搞明白这个-1和-2的意义
        # mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        """
        init weight value using uniform distribution
        :param w:
        :param c:
        :param irange: default 1e-5
        :return:
        """
        # 没有用到这个函数，不用着急翻译
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(axis=-1, keepdims=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v