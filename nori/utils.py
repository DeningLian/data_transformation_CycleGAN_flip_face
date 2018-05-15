import numpy as np
import torch 
from torch.autograd import Variable
from torch import Tensor
def make_input(x, cuda=True, requires_grad=False):
    ## make x be float32
    if x is None: return x
    x = Tensor(np.array(x, dtype='float32'))
    if cuda and torch.cuda.is_available(): x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

MI = make_input