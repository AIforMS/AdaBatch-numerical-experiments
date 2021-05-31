'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
from __future__ import print_function

import argparse
import os
import shutil
import numpy as np
import math
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

import torch.distributed as dist
import torch.utils.data.distributed

from timeit import default_timer as timer

# global timers
fp_tot = 0.
bp_tot = 0.
zero_grad_freq = 1

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

print(f"model_names are: {model_names}")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40, 60, 80],
                    help='Decrease learning rate at these epochs.')

parser.add_argument('--gamma', type=float, default=0.375, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default=os.path.expanduser('~/luliyun/code/AdaBatch/checkpoint/vgg19/dis'),
                    type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--savez', default=os.path.expanduser('~/luliyun/code/AdaBatch/checkpoint/数值/dis'), type=str,
                    metavar="PATH", help='path to save 数值')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, default=20, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

########## adaptive batch resize factor (1-fixed; k-adaptive, increasing by k: 2)
parser.add_argument('--resize-factor', type=float, default=1, metavar='FACTOR',
                    help='Increases the batch size by a factor of FACTOR.')
parser.add_argument('--resize-freq', type=int, default=20, metavar='FREQ',
                    help='Batch sizes is increased if training loss does not improve for FREQ consecutive epochs.')
parser.add_argument('--warmup', type=int, default=0, metavar='N', help='Number of epochs to warmup the learning rate.')
parser.add_argument('--baseline-batch', type=int, default=128, metavar='N',
                    help='Baseline batch size used to compute learning rate scaling during warmup.')
parser.add_argument('--zero-grad-freq', type=int, default=1, metavar='N', help='frequency that gradients are zeroed.')

# Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Distributed train
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--dist-url', default='file:///public/home/jd_yangfeng/luliyun/code/AdaBatch/share/1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--world-size', default=3, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='the rank of GPU')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
print(f"++++++++++is cuda avail: {use_cuda} ++++++++++")

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print('Random Seed is %d' % (args.manualSeed))
# random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.cuda.manual_seed(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc, zero_grad_freq
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    zero_grad_freq = args.zero_grad_freq

    # 1、init Message communication：tcp/ip or shared file
    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # 2、load data, divide the data into several samples
    print('==> Preparing dataset %s' % args.dataset)
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
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    # print(len(trainset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                                        num_replicas=args.world_size,
                                                                        rank=args.rank)
    else:
        train_sampler = None


    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False,
                                  num_workers=args.workers,
                                  sampler=train_sampler)
    print(len(trainloader))

    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            cardinality=args.cardinality,
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop,
        )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
        )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    # 3、build the DDP model
    if args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    multisteplr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)  # lr 衰减策略

    # Resume
    title = 'cifar-100-' + args.arch

    adjusted_lr = args.lr
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    new_batch_size = args.train_batch

    # Train and val
    mb_epoch = 0

    test_tot = 0.
    train_tot = 0.
    resize_tot = 0.
    torch.cuda.synchronize()
    rt_start = timer()
    prev_loss = -1
    error = 1
    loss5 = []
    npz_save = {
        'loss': [],
        'acc': [],
        'epoch': args.epochs,
        'batch_size': [],
        'lr': []
    }
    fix_ada = ''

    min_loss = float('inf')
    reset_cnt = 0
    lrstep = 0
    warmup_lr = 0
    if (args.warmup != 0):
        lrstep = ((args.lr * (args.train_batch / args.baseline_batch)) - args.lr) / (args.warmup - 1)
        warmup_lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        # set the epoch for sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch)

        torch.cuda.synchronize()
        train_start = timer()
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        torch.cuda.synchronize()
        train_tot += timer() - train_start

        multisteplr.step()
        lr_ = optimizer.state_dict()['param_groups'][0]['lr']

        test_start = timer()
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        torch.cuda.synchronize()
        test_tot += timer() - test_start
        print('Epoch: [%d | %d] LR: %f training loss: %f test accuracy: %f' % (
        epoch + 1, args.epochs, lr_, train_loss, test_acc))

        npz_save['loss'].append(train_loss)
        npz_save['acc'].append(test_acc)
        npz_save['lr'].append(lr_)

        if (epoch + 1) < args.warmup:
            warmup_lr += lrstep
            adjusted_lr = warmup_lr
            # reset_learning_rate(optimizer, warmup_lr)

        if train_loss < min_loss:
            min_loss = train_loss
            reset_cnt = 0
        else:
            reset_cnt += 1
        resize_start = timer()
        # if(reset_cnt == args.resize_freq):
        if (args.resize_freq != 0 and ((epoch + 1) % args.resize_freq == 0)):
            # args.resize_freq = args.resize_freq*2
            min_loss = float('inf')
            reset_cnt = 0
            additional_decay = 1.
            if (zero_grad_freq == 1):
                new_batch_size = math.floor(new_batch_size * args.resize_factor)
                trainloader = data.DataLoader(trainset, batch_size=new_batch_size, shuffle=False, drop_last=True,
                                              num_workers=args.workers, sampler=train_sampler)
                print('Increasing batch size to %d with LR: %f' % (new_batch_size, lr_))
            else:
                zero_grad_freq *= args.resize_factor
                if (zero_grad_freq * args.train_batch) > len(trainset):
                    additional_decay = (zero_grad_freq * args.train_batch)
                    zero_grad_freq = math.floor(len(trainset) / args.train_batch)
                    additional_decay = (zero_grad_freq * args.train_batch) / additional_decay
            # if zero_grad_freq*new_batch_size > len(trainset):
            #    additional_decay = (len(trainset)/(zero_grad_freq*new_batch_size))
            #    zero_grad_freq = 1
            #    new_batch_size = len(trainset)
            adjusted_lr *= args.gamma * additional_decay
            # reset_learning_rate(optimizer, adjusted_lr)
            print('Increasing batch size to %d with LR: %f' % (new_batch_size * zero_grad_freq, lr_))
            # half_learning_rate(optimizer, args.gamma)
            # trainloader = data.DataLoader(trainset, batch_size=new_batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
            # print ('Doubling batch size to %d with LR: %f' % (new_batch_size, state['lr']))
            npz_save['batch_size'].append(new_batch_size)
        mb_epoch += 1
        torch.cuda.synchronize()
        resize_tot += timer() - resize_start

    # save model
    if args.resize_factor == 1:
        checkpoint_name = 'fixed_' + str(args.epochs) + '.pth'
        fix_ada = 'fixed'
    else:
        checkpoint_name = 'adabatch_' + str(args.epochs) + '.pth'
        fix_ada = 'ada'

    if args.rank == 0:
        save_checkpoint(model.state_dict(), is_best=False, checkpoint=args.checkpoint, filename=checkpoint_name)
        print('save checkpoint sucessfully')

    torch.cuda.synchronize()
    rt_tot = timer() - rt_start
    print('Best acc:')
    print(best_acc)

    print('\nTotal Forward Prop. Time: %.3f s' % (fp_tot))
    print('Total Backward Prop. Time: %.3f s' % (bp_tot))
    print('Total Training Time: %.3f s' % (train_tot))
    print('Total Test Time: %.3f s' % (test_tot))
    print('Total Batch Resize Time: %.3f s' % (resize_tot))
    print('Total Running Time: %.3f s' % (rt_tot))

    np.savez(os.path.join(args.savez, f'{args.arch}数值{fix_ada}{args.train_batch}rank{args.rank}#.npz'), loss=npz_save['loss'],
             acc=npz_save['acc'], epoch=args.epochs,
             FP_time=fp_tot, BP_time=bp_tot, train_time=train_tot, test_time=test_tot, batch_resize_time=resize_tot,
             running_time=rt_tot, best_acc=best_acc, batch_size=npz_save['batch_size'], lr=npz_save['lr'])
    print('save npz succsesfully')


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    global fp_tot, bp_tot
    global zero_grad_freq
    correct = 0
    total = 0
    batches_processed = 0
    accum_loss = 0
    stored_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        torch.cuda.synchronize()
        fp_start = timer()
        outputs = model(inputs)
        torch.cuda.synchronize()
        fp_tot += timer() - fp_start

        loss = criterion(outputs, targets) / zero_grad_freq

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cuda() == targets.data.cuda()).sum()
        total += targets.size(0)

        # compute gradient and do SGD step
        if ((batch_idx) % (zero_grad_freq)) == 0:
            accum_loss = 0
            optimizer.zero_grad()

        torch.cuda.synchronize()

        bp_start = timer()
        loss.backward()
        accum_loss += loss.item()
        torch.cuda.synchronize()
        bp_tot += timer() - bp_start
        if ((batch_idx + 1) % zero_grad_freq) == 0:
            optimizer.step()
            stored_loss = accum_loss
            batches_processed += 1
        # model.update_loss(loss.data[0])
    return (stored_loss, float(correct) / total)


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    correct = 0
    total = 0
    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cuda() == targets.data.cuda()).sum()
        total += targets.size(0)
    acc = 100 * float(correct) / total
    best_acc = max(best_acc, acc)
    return (loss.item(), acc)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


def reset_learning_rate(optimizer, lr):
    global state
    state['lr'] = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


def half_learning_rate(optimizer, factor):
    global state
    state['lr'] *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    print(os.getcwd())
    # print(os.path.expanduser('~/fromgithub/AdaBatch/checkpoint/cifar100'))
    main()
