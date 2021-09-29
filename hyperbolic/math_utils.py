import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np

eps = 1e-15

def artanh(x):
    art = Artanh()
    return art.construct(x)

class Artanh(nn.Cell):
    def __init__(self):
        super(Artanh, self).__init__()

    def construct(self, x):
        x = np.clip(x, -1+eps, 1-eps)
        sub = ops.Sub()
        sub(np.log(1 + x), np.log(1 - x))
        out = 0.5 * sub(np.log(1 + x), np.log(1 - x))
        return out
