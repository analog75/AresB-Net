from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import gc
import torch
import argparse
import data
import util
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle # added for model clone
from torchstat import stat
from models import *
from torch.autograd import Variable
from torchsummary import summary

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
#suppress all warnings
import warnings
warnings.filterwarnings("ignore")


class AverageMeter(object):
#{{{
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
#}}}

def accuracy(output, target, topk=(1,)):
#{{{
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
#}}}

def save_state(model, best_acc, trainedfilename):
#{{{
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    #torch.save(state, 'models/nin.pth.tar')
    torch.save(state, trainedfilename)
#}}}

def train(train_loader, model, criterion, optimizer, epoch, args):
#{{{
    outputfile_handler =open(args.outputfile + "train", 'a+')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
   
      
    # switch to train mode
    model.train()
    bin_gates = [p for p in model.parameters() if getattr(p, 'bin_gate', False)] 
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # process the weights including binarization
        bin_op.binarization()

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()
        for p in bin_gates:
            p.data.clamp_(min=0, max=1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

        #gc.collect()

    print('Epoch: [{0}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Data {data_time.avg:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}'.format(
           epoch, batch_time=batch_time,
           data_time=data_time, loss=losses, top1=top1, top5=top5), file=outputfile_handler)

    print('Loss {loss.avg:.4f} * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}' .format(loss=losses, top1=top1, top5=top5))

#}}}

def test(val_loader, model, criterion, args, epoch=0):
#{{{
    print('==>tested using AresB-Net')
    outputfile_handler =open(args.outputfile + "test", 'a+')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    bin_op.binarization()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    bin_op.restore()

    print('Epoch: [{0}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}'.format(
           epoch, batch_time=batch_time, loss=losses, top1=top1, top5=top5), 
           file=outputfile_handler)

    print('Loss {loss.avg:.4f} * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(
           loss=losses, top1=top1, top5=top5))

    return top1.avg
#}}}

def adjust_learning_rate(optimizer, epoch):
    update_list = [160, 200, 260, 320]
#{{{
#    update_list = [120, 200, 260, 320]
#    update_list = [150, 250, 350]
#    update_list = [120, 200, 260, 320]
#    update_list = [150, 250, 350]
#    update_list = [150, 250, 320]
#    update_list = [15, 25, 35]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    return
#}}}

if __name__=='__main__':
    # prepare the options
    #{{{
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', action='store', default='resnet18',
            help='default architecture for the network: resnet18')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--epochs', default=360, type=int, metavar='N',
            help='number of total epochs to run')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--lr', action='store', default='0.1', type=float,
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--outputfile', action='store', default='test.out',
            help='output file')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', 
            help='frequence of printing data(default: 100)')
    parser.add_argument('--trainedfile', action='store', default='train/test.pth.tar',
            help='trained file')
    parser.add_argument('--sr', action='store_true',
            help='training using input with stochastic rounding')
    parser.add_argument('--wd', '--weight-decay', default=0.00001, type=float,
            metavar='W', help='weight decay (default: 1e-5)',
            dest='weight_decay')
    #}}}

    args = parser.parse_args()
    print('==> Options:',args)
    if os.path.isfile(args.outputfile+"train"):
      os.remove(args.outputfile+"train")
    if os.path.isfile(args.outputfile+"test"):
      os.remove(args.outputfile+"test")
    if os.path.isfile(args.trainedfile):
      os.remove(args.trainedfile)
    #outputfile_handler =open(args.outputfile, 'w')

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
            shuffle=False, num_workers=6)

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'aresbnet10':
      model = AresBNet10()
    elif args.arch == 'aresbnet18':
      model = AresBNet18()
    elif args.arch == 'aresbnet34':
      model = AresBNet34()
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained, map_location='cuda:0')
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
      prec1 = test(testloader, model, criterion, args)
      exit(0)

    for epoch in range(0, args.epochs):
      adjust_learning_rate(optimizer, epoch)
      train(trainloader, model, criterion, optimizer, epoch, args)

      # evaluate on test set
      prec1 = test(testloader, model, criterion, args, epoch)

      if epoch == args.epochs -1:
        save_state(model, prec1, args.trainedfile)
    exit(0)


