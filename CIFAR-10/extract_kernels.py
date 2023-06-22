'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
import math
import random

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from resnet_learn import *
from utils import progress_bar
from ir_1w1a_learn import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--log', type=str, default='./log/log.txt', help='path and name of the log file')
parser.add_argument('--EC', default=0.5, type=float,
                    help='expected connections')
parser.add_argument('--gamma', default=0.0, type=float,
                    help='strength of the regularizer (0<=gamma<=1)') 
parser.add_argument('--batch_size', type=int, default=128, help='batch size')           
parser.add_argument('--num_workers', type=int, default=2, help='CPU workers for the Dataloader')                   
parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
original_stdout = sys.stdout


if not os.path.exists('log'):
    os.mkdir('log')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')


param_string= '_EC' + str(round(args.EC*100)) + '_gamma' + str(round(args.gamma*1e6)) +'_batch' + str(args.batch_size) + '_epochs' + str(args.epochs) + '_lr' + str(round(args.lr*1e6)) + '_alpha' + str(round(args.alpha*1e2))
log_string = './log/log_learn' + param_string + '.txt'
ckpt_string = './checkpoint/ckpt_learn' + param_string + '.pth'
ckpt_best_string = './checkpoint/ckpt_learn' + param_string + '_best.pth'
output_kernels = './log/kernels' + param_string + '.txt'

# Data
print('==> Preparing data..')
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(ckpt_best_string,map_location = device)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

use_cuda = torch.cuda.is_available()
net.eval()
count = 0
with torch.no_grad():
    for module in net.modules():
        if isinstance(module,IRConv2d):
            count += 1
    L = count			
    kernels = torch.zeros((L,512))
    pruned_filters = torch.zeros((L))
    all_filters = torch.zeros((L))
    count = 0
    for module in net.modules():
        if isinstance(module,IRConv2d):
            tmp = 1.0*(module.weight > 0.0)
            tmp_kernels=torch.sum(tmp.view(tmp.size(0),tmp.size(1),tmp.size(2)*tmp.size(3))*torch.Tensor([1.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0]).cuda(),dim=2)
            tmp_kernels = tmp_kernels.type(torch.int64)
            tmp_kernels = tmp_kernels.view((torch.numel(tmp_kernels)))
            tmp_kernels = torch.bincount(tmp_kernels)
            kernels[count][0:torch.numel(tmp_kernels)] = tmp_kernels
            count += 1
			
            with open(output_kernels, 'a') as f:
                sys.stdout = f
                print('kernels(%d,:)=['%(count),end='')
                for i in range(511):
                    print('%d,'%(kernels[count-1][i]),end='')
                print('%d];'%(kernels[count-1][511]))
                sys.stdout = original_stdout
    count = 0
    for module in net.modules():
        if isinstance(module,IRConv2d):
            tmp = 1.0*(module.weight > 0.0)
            tmp_kernels=torch.sum(tmp.view(tmp.size(0),tmp.size(1),tmp.size(2)*tmp.size(3))*torch.Tensor([1.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0]).cuda(),dim=2)
            pruned_filters[count] = torch.sum(1.0*(torch.sum(tmp_kernels,dim=1)<1.0))
            all_filters[count]=module.weight.size(0)
            count += 1
			
    with open(output_kernels, 'a') as f:
        sys.stdout = f
        print('\npruned_filters=[',end='')
        for i in range(count-1):
            print('%d,'%(pruned_filters[i]),end='')
        print('%d];'%(pruned_filters[count-1]))
        print('all_filters=[',end='')
        for i in range(count-1):
            print('%d,'%(all_filters[i]),end='')
        print('%d];'%(all_filters[count-1]))
        sys.stdout = original_stdout
                
    