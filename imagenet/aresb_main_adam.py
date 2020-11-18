from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import gc
import shutil
import time 
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle # added for model clone
from torchstat import stat
from models import *
from torch.autograd import Variable
from torchsummary import summary
#HJKIM
from datasets import caffelmdb

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Resize
#suppress all warnings
import warnings
warnings.filterwarnings("ignore")

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

#def train(train_loader, model, criterion, optimizer, epoch, args, scheduler):
def train(train_loader, model, criterion, optimizer, epoch, args):
#{{{
    outputfile_handler =open(args.trainout, 'a+')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # process the weights including binarization
        bin_op.binarization()
        #target = target.cuda(async=True)
        #input = input.cuda(non_blocking=True).half()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        #input_var = torch.autograd.Variable(input).half()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #scheduler.step()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()

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

        gc.collect()

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

def validate(val_loader, model, criterion, args, epoch=0):
#{{{
    print('==>tested using AresB-Net')
    outputfile_handler =open(args.valout, 'a+')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bin_op.binarization()
    for i, (input, target) in enumerate(val_loader):
        #target = target.cuda(async=True)
        #input = target.cuda(non_blocking=True).half()
        #input = target.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
          input_var = torch.autograd.Variable(input).cuda()
          target_var = torch.autograd.Variable(target)
          output = model(input_var)

        # compute output
        #output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

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

#def adjust_learning_rate(optimizer, epoch, args):
##{{{
#    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#    lr = args.lr * (0.1 ** (epoch // 30))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr
##}}}

def adjust_learning_rate(optimizer, epoch, args):
    update_list = [40, 50, 65, 80]
    #update_list = [27]
#{{{
#    update_list = [30, 40]
#    update_list = [15, 25, 35]
#    update_list = [150, 250, 350]
#    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return
#}}}

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#{{{
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
#}}}

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
    """Computes the precision@k for the specified values of k"""
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

if __name__=='__main__':   
    # prepare the options
    #{{{
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', action='store', default='resnet18',
            help='default architecture for the network: resnet18')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', 
            help='mini-batch size (default: 256)')
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',                                            
            help='number of total epochs to run')
    parser.add_argument('--lr', action='store', default='0.1', type=float,
            help='the intial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
            help='momentum')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', 
            help='frequence of printing data(default: 100)')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--sr', action='store_true',
            help='training using input with stochastic rounding')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
            help='manual epoch number (useful on restarts)')   
    parser.add_argument('--train', action='store', 
            default='./data/torch_ilsvrc12_train_lmdb', help='path to training dataset')
    parser.add_argument('--trainout', action='store', default='train.out',
            help='training output file')
    parser.add_argument('--trainedfile', action='store', default='train/test.pth.tar',
            help='trained file')
    parser.add_argument('--val', action='store', 
            default='./data/torch_ilsvrc12_val_lmdb', help='path to validation dataset')
    parser.add_argument('--valout', action='store', default='val.out',
            help='training output file')
    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)',
            dest='weight_decay')

    #}}}

    best_prec1 = 0

    args = parser.parse_args()
    print('==> Options:',args)
    if os.path.isfile(args.trainedfile):
      os.remove(args.trainedfile)
    if os.path.isfile(args.trainout):
      os.remove(args.trainout)
    if os.path.isfile(args.valout):
      os.remove(args.valout)

    #outputfile_handler =open(args.outputfile, 'w')

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare the data
#    if not os.path.isfile(args.data+'/train_data'):
#        # check the data path
#        raise Exception\
#                ('Please assign the correct training data path with --data <DATA_PATH>')
#
#    if not os.path.isfile(args.data+'/test_data'):
#        # check the data path
#        raise Exception\
#                ('Please assign the correct test data path with --data <DATA_PATH>')

    # Data
    print('==> Preparing data..')
    # Data loading code
    image_folder = caffelmdb.ImageFolderLMDB

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = image_folder(
        args.train,
        transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(224, scale=(0.466,0.875)),
            #transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            #transforms.RandomErasing(),
        ]))

    val_dataset = image_folder(
        args.val, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

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
#HJKIM
        #for m in model.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.weight.data.normal_(0, 0.05)
                #m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained, map_location='cuda:0')
        best_prec1 = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    #print(model)

    #model.half()
    #for layer in model.modules():
    #  if isinstance(layer, nn.BatchNorm2d):
    #    layer.float()

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

    #HJKIM: This version is for ADAM optimizer
    optimizer = optim.Adam(params, lr=0.10, weight_decay=0.00001)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    #for epoch in range(args.start_epoch):
    #    scheduler.step()
    #optimizer = optim.SGD(params, momentum=args.momentum, 
    #                      weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()


    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
      validate(val_loader, model, criterion, args)
      exit(0)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch, args, scheduler)
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, args, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
          save_state(model, best_prec1, args.trainedfile)

        if epoch == args.epochs -1:
          save_state(model, prec1, args.trainedfile + "final")
    exit(0)
#}}}


