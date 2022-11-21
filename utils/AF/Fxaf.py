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
    # x = torch.tanh(-1+(1/(0.1 + torch.exp(-input))))
    # x = torch.log(1+torch.sigmoid(input))# ori good
    x = torch.log(0.25+(1/( 0.25+torch.exp(-input))))# ori good
    # x =(-1+(1/(torch.exp(-input)))) # sigmoid modefied
    # np.clip(fx+0.25,-0.5,0.9)
    # x = torch.clip(x+0.2,-0.5,0.9) # this help ti thin the edge check 2
    # adding 0.25
    # x = torch.clip(torch.tanh(x)+0.25,-0.5,0.9)
    # x = torch.clip(x,-0.5,0.9)
    return input * x