import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
import numpy as np
import os
import pdb
import argparse
import logging
import time

parser = argparse.ArgumentParser(description="test_acc", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--epoch", type=str)
args = parser.parse_args()

torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

log = logging.getLogger("mylog")
formatter = logging.Formatter("%(asctime)s : %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(streamHandler)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_test = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

if args.model == "resnet18":
    net = models.resnet18()
elif args.model == "resnet50":
    net = models.resnet50()

net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

if args.dataset == "cifar10":
    test_data = dset.CIFAR10('../data/cifar10', train=False, transform=transform_test)
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=10, bias=True)
    if args.model == "resnet18":
        if args.epoch == "88":
            net.load_state_dict(torch.load('./ckpt/resnet18_cifar10_epoch88.pt'))
        elif args.epoch == "147":
            net.load_state_dict(torch.load('./ckpt/resnet18_cifar10_epoch147.pt'))
    elif args.model == "resnet50":
        if args.epoch == "121":
            net.load_state_dict(torch.load('./ckpt/resnet50_cifar10_epoch121.pt'))
        elif args.epoch == "124":
            net.load_state_dict(torch.load('./ckpt/resnet50_cifar10_epoch124.pt'))
elif args.dataset == "cifar100":
    test_data = dset.CIFAR100('../data/cifar100', train=False, transform=transform_test)
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=100, bias=True)
    if args.model == "resnet18":
        net.load_state_dict(torch.load('./ckpt/resnet18_cifar100_epoch109.pt'))
    elif args.model == "resnet50":
        net.load_state_dict(torch.load('./ckpt/resnet50_cifar100_epoch150.pt'))

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=128, shuffle=False
)

if torch.cuda.is_available():
    net.cuda()

def test(epoch):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = net(data)
            loss = F.cross_entropy(output, target)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    accuracy = correct / len(test_loader.dataset)
    log.debug("accuracy:{0:5f}".format(accuracy))

for epoch in range(10):
    test(epoch)