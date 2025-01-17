# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.nn as nn

from models.wrn import WideResNet
import torchvision.models as models
import utils.utils_awp as awp
from utils.display_results import  get_measures, print_measures
from utils.validation_dataset import validation_split

import os
import logging
import pdb

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with DOE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')

parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--begin_epoch', type=int, default=0)

parser.add_argument('--model', type=str, choices=['wrn', 'resnet18', 'resnet50'])
parser.add_argument('--lr_adjust', type=str, choices=['cos', 'poem'])
parser.add_argument('--oe_setting', type=str, help='l + 0.5 * loe')

torch.cuda.set_device(0) # 选择第0张卡
# torch.cuda.set_device(1) # 选择第1张卡

args = parser.parse_args()
torch.manual_seed(1)
np.random.seed(args.seed)
torch.cuda.manual_seed(1)

if args.oe_setting == "yes":
    log = logging.getLogger(args.dataset + "_" + args.model + "_" + str(args.learning_rate) + "_" + args.lr_adjust + "_oe" + ".log")
    fileHandler = logging.FileHandler(args.dataset + "_" + args.model + "_" + str(args.learning_rate) + "_" + args.lr_adjust + "_oe" + ".log", mode='a')
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.debug(args.dataset + "_" + args.model + "_" + str(args.learning_rate) + "_" + args.lr_adjust + "_oe" + ".log")
    log.debug("")
else:
    log = logging.getLogger(args.dataset + "_" + args.model + "_" + str(args.learning_rate) + "_" + args.lr_adjust + ".log")
    fileHandler = logging.FileHandler(args.dataset + "_" + args.model + "_" + str(args.learning_rate) + "_" + args.lr_adjust + ".log", mode='a')
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.debug(args.dataset + "_" + args.model + "_" + str(args.learning_rate) + "_" + args.lr_adjust + ".log")
    log.debug("")
for k, v in args._get_kwargs():
    log.debug(str(k)+": "+str(v))
log.debug("")

cudnn.benchmark = True  # fire on all cylinders

# mean and standard deviation of channels of CIFAR-10 images
if 'cifar' in args.dataset:
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
else: 
    mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()


train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10('../data/cifar10', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10('../data/cifar10', train=False, transform=test_transform)
    cifar_data = dset.CIFAR100('../data/cifar100', train=False, transform=test_transform, download=True) 
    num_classes = 10
else:
    train_data_in = dset.CIFAR100('../data/cifar100', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100('../data/cifar100', train=False, transform=test_transform)
    cifar_data = dset.CIFAR10('../data/cifar10', train=False, transform=test_transform, download=True)
    num_classes = 100
calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'
ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)])) # test(3)

train_loader_in = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=False)
train_loader_out = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=False)

texture_data = dset.ImageFolder(root="../data/dtd/images", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
places365_data = dset.ImageFolder(root="../data/places365", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
isun_data = dset.ImageFolder(root="../data/iSUN",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
cifar_loader = torch.utils.data.DataLoader(cifar_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.cuda(), target.cuda()
            output = net(data)
            # smax = to_np(F.softmax(output, dim=1))
            smax = to_np(output) # DOE中使用的score
            _score.append(-np.max(smax, axis=1)) # DOE中使用的score
    if in_dist:
        return concat(_score).copy() # , concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(mylog, ood_loader, in_score, num_to_avg=args.num_to_avg):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(mylog, auroc, aupr, fpr, '')
    return fpr, auroc, aupr

def train(epoch, diff):
    if args.lr_adjust == 'poem':
        adjust_learning_rate(optimizer, epoch) # 修改

    if args.model == 'wrn':
        proxy = WideResNet(args.layers, num_classes, args.widen_factor, dropRate = 0).cuda()
    elif args.model == 'resnet18':
        proxy = models.resnet18()
        proxy.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        proxy.fc = torch.nn.Linear(in_features=proxy.fc.in_features, out_features=num_classes, bias=True)
        proxy.cuda()
    elif args.model == 'resnet50':
        proxy = models.resnet50()
        proxy.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        proxy.fc = torch.nn.Linear(in_features=proxy.fc.in_features, out_features=num_classes, bias=True)
        proxy.cuda()

    proxy_optim = torch.optim.SGD(proxy.parameters(), lr=1) # 辅助模型

    net.train()

    loss_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_in.dataset))
    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        data, target = torch.cat((in_set[0], out_set[0]), 0), in_set[1]
        data, target = data.cuda(), target.cuda()

        if epoch >= args.warmup:
            if args.dataset == 'cifar10':
                gamma =  torch.Tensor([1e-1,1e-2,1e-3,1e-4])[torch.randperm(4)][0]
            else: 
                gamma =  torch.Tensor([1e-1,1e-2,1e-3,1e-4])[torch.randperm(4)][0] # 31
            proxy.load_state_dict(net.state_dict())
            proxy.train()
            scale = torch.Tensor([1]).cuda().requires_grad_()
            x = proxy(data) * scale
            l_sur = (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean() # 这里求的是min Loe, 所以会差一个负号
            # l_sur = - (x.log_softmax(1) * (x / 0.1).softmax(1).detach()).sum(-1).mean()
            # emb = proxy.intermediate_forward_simple(data) # 修改!
            # reg_sur = torch.sum(torch.autograd.grad(l_sur, [scale], create_graph = True)[0] ** 2) + torch.norm(emb, 1) # 修改! 计算l_sur关于scale的梯度，并将其平方后求和
            reg_sur = torch.sum(torch.autograd.grad(l_sur, [scale], create_graph = True)[0] ** 2)
            proxy_optim.zero_grad()
            reg_sur.backward()
            # l_sur.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1) # norm操作
            proxy_optim.step()
            if epoch == args.warmup and batch_idx == 0:
                diff = awp.diff_in_weights(net, proxy) # 微分操作(第一次还无法加权平均)
                #结合DRO
                # diff upgrade
                # diff_optim = torch.optim.SGD(diff, lr=1)
                # m_phi = - ( - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean() - gamma * diff.norm())
                # m_phi.backward()
                # diff.step()
                #结合DRO
            else:
                diff = awp.average_diff(diff, awp.diff_in_weights(net, proxy), beta = .6) # 指数加权平均操作
                #结合DRO
                # diff upgrade
                # diff_optim = torch.optim.SGD(diff, lr=1)
                # m_phi = - ( - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean() - gamma * diff.norm())
                # m_phi.backward()
                # diff.step()
                #结合DRO

            awp.add_into_weights(net, diff, coeff = gamma)

        #结合DRO
        # gamma upgrade
        # gamma_optim = torch.optim.SGD(gamma, lr=1)
        # func = torch.Tensor([1]).cuda().requires_grad_() - diff
        # func.backward()
        # gamma_optim.step()
        #结合DRO

        # pdb.set_trace()
        x = net(data)
        l_ce = F.cross_entropy(x[:len(in_set[0])], target)
        if args.oe_setting == "yes":
            l_oe = 0.5 * - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean() # 修改
        else:
            l_oe = - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
        if args.dataset == 'cifar10':
            if epoch >= args.warmup:
                loss = l_oe
            else: 
                loss = l_ce +  l_oe
        else: 
            if epoch >= args.warmup:
                loss = l_oe
            else: 
                loss = l_ce +  l_oe
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1) # norm操作
        optimizer.step()

        if epoch >= args.warmup:
            awp.add_into_weights(net, diff, coeff = - gamma)
            optimizer.zero_grad()
            x = net(data)
            l_ce = F.cross_entropy(x[:len(in_set[0])], target)
            loss = l_ce # + l_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1) # norm操作
            optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        sys.stdout.write('\r epoch %2d %d/%d loss %.2f' %(epoch, batch_idx + 1, len(train_loader_in), loss_avg))
        if args.lr_adjust == 'cos':
            scheduler.step()

    print()
    return diff

def test():
    net.eval()
    correct = 0
    y, c = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    return correct / len(test_loader.dataset) * 100

print('Beginning Training\n')
# Restore model
if args.dataset == 'cifar10':
    if args.model == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
        model_path = './ckpt/cifar10_wrn_pretrained_epoch_99.pt'
    elif args.model == 'resnet18':
        net = models.resnet18()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=num_classes, bias=True)
        model_path = './ckpt/resnet18_cifar10_epoch147.pt'
        net.cuda()
    elif args.model == 'resnet50':
        net = models.resnet50()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=num_classes, bias=True)
        model_path = './ckpt/resnet50_cifar10_epoch124.pt'
        net.cuda()
elif args.dataset == 'cifar100':
    if args.model == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
        model_path = './ckpt/cifar100_wrn_pretrained_epoch_99.pt'
    elif args.model == 'resnet18':
        net = models.resnet18()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=num_classes, bias=True)
        model_path = './ckpt/resnet18_cifar100_epoch109.pt'
        net.cuda()
    elif args.model == 'resnet50':
        net = models.resnet50()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=num_classes, bias=True)
        model_path = './ckpt/resnet50_cifar100_epoch170.pt'
        net.cuda()

net.load_state_dict(torch.load(model_path))
optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def adjust_learning_rate(optimizer, epoch, lr_schedule=[4, 6, 8]): # POEM中使用的学习率调整方法
    lr = args.learning_rate
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if args.lr_adjust == 'cos':
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate))
diff = None

def evaluate():
    net.eval()
    print('  FPR95 AUROC AUPR (acc %.2f)' % test())
    log.debug('  FPR95 AUROC AUPR (acc %.2f)' % test())
    in_score = get_ood_scores(test_loader, in_dist=True)
    metric_ll = []
    print('lsun')
    log.debug('lsun')
    metric_ll.append(get_and_print_results(log, lsunc_loader, in_score))
    print('isun')
    log.debug('isun')
    metric_ll.append(get_and_print_results(log, isun_loader, in_score))
    print('texture')
    log.debug('texture')
    metric_ll.append(get_and_print_results(log, texture_loader, in_score))
    print('places')
    log.debug('places')
    metric_ll.append(get_and_print_results(log, places365_loader, in_score))
    print('total')
    log.debug('total')
    print('& %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))
    log.debug('& %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))
    print('cifar')
    log.debug('cifar')
    get_and_print_results(log, cifar_loader, in_score)

# for epoch in range(args.begin_epoch, args.epochs): # 修改
for epoch in range(args.begin_epoch, args.epochs - 1):
    diff = train(epoch, diff)
    # evaluate()

evaluate()
