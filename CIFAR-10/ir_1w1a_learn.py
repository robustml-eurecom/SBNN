import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import binaryfunction
import torch
import math

class sign_function(torch.autograd.Function) :

  @staticmethod
  def forward(ctx, weight):
    return 2.0*(weight>0.0)-1.0
    
  @staticmethod
  def backward(ctx, grad_output):
    grad_input=grad_output
    return grad_input

class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.tau = Parameter(torch.tensor(1.0).float().cuda())
        self.phi = Parameter(torch.tensor(0.0).float().cuda())
        self.tot_weights = torch.numel(self.weight)
        self.ones_weights = torch.numel(self.weight)/2

    def forward(self, input):
        my_sign = sign_function.apply
        w = self.weight
        a = input
        self.ones_weights = 0.5*torch.sum(my_sign(w)+1.0)
        self.tot_weights = torch.numel(w)
        bw = binaryfunction.BinaryQuantize().apply(w, self.k, self.t)
        ba = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)

        output = F.conv2d(ba, self.tau*bw+self.phi, None,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output
