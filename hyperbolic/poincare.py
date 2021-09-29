import numpy
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.numpy as np
from hyperbolic.math_utils import *
from mindspore import Parameter
from mindspore.common import dtype as mstype

def clamp_tanh(x, clamp=15):
    tanh = nn.Tanh()
    return tanh(np.clip(x, -clamp, clamp))

class ManifoldParameter(Parameter):

    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c, name):
        self.c = c
        self.manifold = manifold
        self.name = name

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()

class PoincareBall(object):
    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {mindspore.float32: 4e-3, mindspore.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            np.norm(sqrt_c * self.mobius_add(-p1, p2, c, dim=-1), axis=-1)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        Pow = ops.Pow()
        Rsum = ops.ReduceSum(keep_dims=True)
        x_sqnorm = Rsum(Pow(x, 2), -1)
        return 2 / np.clip((1. - c * x_sqnorm), self.min_norm, np.inf)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        Pow = ops.Pow()
        dp /= Pow(lambda_p, 2)
        return dp

    def proj(self, x, c):
        norm = np.clip(np.norm(x, axis=-1, keepdims=True), self.min_norm, np.inf)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        projected = x / norm * maxnorm
        cond = norm > maxnorm
        return np.where(cond, projected, x)

    def proj_tan(self, u):
        return u

    def proj_tan0(self, u):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = np.clip(np.norm(u, axis=-1, keepdims=True), self.min_norm, np.inf)
        second_term = (
                clamp_tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = np.clip(np.norm(sub, axis=-1, keepdims=True), self.min_norm, np.inf)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = np.clip(np.norm(u, axis=-1, keepdims=True), self.min_norm, np.inf)
        gamma_1 = clamp_tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = np.clip(np.norm(p, axis=-1, keepdims=True), self.min_norm, np.inf)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        Pow = ops.Pow()
        Rsum = ops.ReduceSum(keep_dims=True)
        x2 = Rsum(Pow(x, 2), dim)
        y2 = Rsum(Pow(y, 2), dim)
        xy = Rsum(x * y, dim)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / np.clip(denom, self.min_norm, np.inf)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = np.clip(np.norm(x, axis=-1, keepdims=True), self.min_norm, np.inf)
        mx = np.matmul(x, m.swapaxes(-1, -2))
        mx_norm = np.clip(np.norm(mx, axis=-1, keepdims=True), self.min_norm, np.inf)
        t1 = artanh(sqrt_c * x_norm)
        t2 = clamp_tanh(mx_norm / x_norm * t1)
        res_c = t2 * mx / (mx_norm * sqrt_c)

        t = (mx == 0).astype(mindspore.uint8).asnumpy()
        cond = numpy.prod(t, -1, keepdims=True)
        zeros = ops.Zeros()
        res_0 = zeros(1, res_c.dtype)
        res = np.where(Tensor(cond), res_0, res_c)
        return res

    def init_weights(self, w, irange=1e-5):
        shape = w.shape
        w = numpy.random.uniform(-irange, irange, shape)
        w = Tensor(w).set_dtype(mstype.float64)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        Pow = ops.Pow()
        Rsum = ops.ReduceSum(keep_dims=True)
        u2 = Rsum(Pow(u, 2), dim)
        v2 = Rsum(Pow(v, 2), dim)
        uv = Rsum(u * v, dim)
        uw = Rsum(u * w, dim)
        vw = Rsum(v * w, dim)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / np.clip(d, self.min_norm, np.inf)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        Rsum = ops.ReduceSum(keep_dims=keepdim)
        return Rsum(lambda_x ** 2 * (u * v), -1)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def to_hyperboloid(self, x, c):
        K = 1.0 / c
        sqrtK = K ** 0.5
        sqnorm = np.norm(x, axis=-1, keepdims=True) ** 2
        cat = ops.Concat(-1)
        sqrtK * cat(K + sqnorm, sqrtK * x) / (K - sqnorm)
        return sqrtK * cat((K + sqnorm, sqrtK * x)) / (K - sqnorm)

    def klein_constraint(self, x):
        last_dim_val = x.shape[-1]
        norm = np.norm(x, axis=-1).reshape(-1, 1)
        maxnorm = (1 - self.eps[x.dtype])
        cond = norm > maxnorm
        x_reshape = x.reshape(-1, last_dim_val)
        projected = x_reshape / (norm + self.min_norm) * maxnorm
        x_reshape = np.where(cond, projected, x_reshape)
        x = x_reshape.reshape(x.shape)
        return x

    def to_klein(self, x, c):
        Rsum = ops.ReduceSum(keep_dims=True)
        x_2 = Rsum(x * x, -1)
        x_klein = 2 * x / (1.0 + x_2)
        x_klein = self.klein_constraint(x_klein)
        return x_klein

    def klein_to_poincare(self, x, c):
        sqrt = ops.Sqrt()
        Rsum = ops.ReduceSum(keep_dims=True)
        x_poincare = x / (1.0 + sqrt(1.0 - Rsum(x * x, -1)))
        print(x_poincare)
        x_poincare = self.proj(x_poincare, c)
        print(x_poincare)
        return x_poincare

    def lorentz_factors(self, x):
        x_norm = np.norm(x, axis=-1)
        return 1.0 / (1.0 - x_norm ** 2 + self.min_norm)

    def einstein_midpoint(self, x, c):
        expand_dims = ops.ExpandDims()
        Rsum = ops.ReduceSum(keep_dims=True)
        x = self.to_klein(x, c)
        x_lorentz = self.lorentz_factors(x)
        x_norm = np.norm(x, axis=-1)
        # deal with pad value
        x_lorentz = (1.0 - (x_norm == 0).astype(mindspore.float64)) * x_lorentz
        x_lorentz_sum = Rsum(x_lorentz, -1)
        x_lorentz_expand = expand_dims(x_lorentz, -1)
        x_midpoint = Rsum(x_lorentz_expand * x, -1) / x_lorentz_sum
        x_midpoint = self.klein_constraint(x_midpoint)
        x_p = self.klein_to_poincare(x_midpoint, c)
        return x_p


