# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

from models.wrn import WideResNet
import utils.utils_awp as awp
from utils.display_results import  get_measures, print_measures
from utils.validation_dataset import validation_split

import os
import logging

parser = argparse.ArgumentParser(description='Test MSP', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str, help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')

torch.cuda.set_device(0) # 选择第0张卡

args = parser.parse_args()
torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)

log = logging.getLogger("msp_test.log")
fileHandler = logging.FileHandler("./logging/msp_test.log", mode='a')
log.setLevel(logging.DEBUG)
log.addHandler(fileHandler)
log.debug("msp_test.log")
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

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    test_data = dset.CIFAR10('../data/cifar10', train=False, transform=test_transform)
    cifar_data = dset.CIFAR100('../data/cifar100', train=False, transform=test_transform) 
else:
    test_data = dset.CIFAR100('../data/cifar100', train=False, transform=test_transform)
    cifar_data = dset.CIFAR10('../data/cifar10', train=False, transform=test_transform)

ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)])) # test(3)

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
            smax = to_np(output)
            _score.append(-np.max(smax, axis=1)) # DOE中使用的score
    if in_dist:
        return concat(_score).copy() # , concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def msp_get_ood_scores(loader, in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.cuda(), target.cuda()
            output = net(data)
            # smax = to_np(output)
            smax = to_np(F.softmax(output, dim=1))
            _score.append(-np.max(smax, axis=1)) # DOE中使用的score
    if in_dist:
        return concat(_score).copy() # , concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(mylog, ood_loader, in_score, num_to_avg=args.num_to_avg):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = msp_get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(mylog, auroc, aupr, fpr, '')
    return fpr, auroc, aupr

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

print('Beginning\n')
if args.dataset == 'cifar10':
    if args.model == "resnet18":
        net = models.resnet18()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=10, bias=True)
        net.load_state_dict(torch.load('./ckpt/resnet18_cifar10_epoch147.pt'))
    elif args.model == "resnet50":
        net = models.resnet50()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=10, bias=True)
        net.load_state_dict(torch.load('./ckpt/resnet50_cifar10_epoch124.pt'))
elif args.dataset == 'cifar100':
    if args.model == "resnet18":
        net = models.resnet18()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=100, bias=True)
        net.load_state_dict(torch.load('./ckpt/resnet18_cifar100_epoch109.pt'))
    elif args.model == "resnet50":
        net = models.resnet50()
        net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=100, bias=True)
        net.load_state_dict(torch.load('./ckpt/resnet50_cifar100_epoch150.pt'))
net.cuda()

net.eval()
print('  FPR95 AUROC AUPR (acc %.2f)' % test())
log.debug('  FPR95 AUROC AUPR (acc %.2f)' % test())
in_score = msp_get_ood_scores(test_loader, in_dist=True)
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
