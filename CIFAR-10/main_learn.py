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


# Data
print('==> Preparing data..')
with open(log_string, 'a') as f:
    sys.stdout = f
    print('args %s'%(args))
    print('==> Preparing data..')
    sys.stdout = original_stdout
    
        
    
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
with open(log_string, 'a') as f:
    sys.stdout = f
    print('==> Building model..')
    sys.stdout = original_stdout

net = ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    with open(log_string, 'a') as f:
        sys.stdout = f
        print('==> Resuming from checkpoint..')
        sys.stdout = original_stdout
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(ckpt_string)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(),
            lr=args.lr,weight_decay=0)            
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def Log_UP(K_min, K_max, epoch):
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float().cuda()

use_cuda = torch.cuda.is_available()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    with open(log_string, 'a') as f:
        sys.stdout = f
        print('\nEpoch: %d' % epoch)
        sys.stdout = original_stdout
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    h = 0
    total_params = 0;
    T_min, T_max = 1e-1, 1e1
    t = Log_UP(T_min, T_max, epoch)
    if (t < 1):
        k = 1 / t
    else:
        k = torch.tensor([1]).float().cuda()
        
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module,IRConv2d):
                total_params = module.tot_weights if total_params==0 else total_params+module.tot_weights
                module.k = k
                module.t = t
    total_params = torch.tensor(total_params).float().cuda()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        total_1s = torch.tensor([0.0]).float().cuda();

        optimizer.zero_grad()
        p_dist=torch.FloatTensor([0.0, 0.0]).cuda();
        outputs = net(inputs)
        
        #loss = criterion(outputs, targets)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        for module in net.modules():
            if isinstance(module,IRConv2d):
                total_1s = module.ones_weights if total_1s==0 else total_1s+module.ones_weights

        h = 0 if args.EC == 0.5 else F.relu(total_1s/total_params - args.EC)
        with torch.no_grad():
            _lambda = 0 if h<=1e-4 else  args.gamma*loss / (h * (1-args.gamma))
        loss += _lambda*h

        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            for module in net.modules():
                if isinstance(module,IRConv2d):
                    module.weight.data=torch.clamp(module.weight.data,min=-1.5,max=1.5)
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        #correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | EC %.2f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*total_1s/total_params, total_1s, total_params))
    with open(log_string, 'a') as f:
        sys.stdout = f
        print('Train: Loss: %.3f | Acc: %.3f%% (%d/%d) | EC %.2f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*total_1s/total_params, total_1s, total_params))
        sys.stdout = original_stdout


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        with open(log_string, 'a') as f:
                sys.stdout = f
                print('Test: Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                sys.stdout = original_stdout

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, ckpt_best_string)
        best_acc = acc

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }     
    torch.save(state, ckpt_string)
    


for epoch in range(start_epoch, args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
