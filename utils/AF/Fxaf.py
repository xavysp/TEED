"""
Check Swish, tanh, mish and smish
"""

import torch


@torch.jit.script
def xaf(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(sigmoid(x))))
    See additional documentation for mish class.
    """
    x = torch.log(1+torch.sigmoid(input))
    # np.clip(fx+0.25,-0.5,0.9)
    x = torch.clip(x+0.25,-0.5,0.9)
    return input * x