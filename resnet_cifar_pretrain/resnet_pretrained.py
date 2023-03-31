import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from cutout import Cutout
import numpy as np
import os
import pdb
import argparse
import logging
import time

parser = argparse.ArgumentParser(description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--best_acc', type=float, default=0.0)
args = parser.parse_args()

torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

log = logging.getLogger("mylog")
formatter = logging.Formatter("%(asctime)s : %(message)s")
fileHandler = logging.FileHandler(os.path.join("./", args.model + "_"+ args.dataset + ".log"), mode="a")
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(fileHandler)
log.addHandler(streamHandler)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_train = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean, std),
    trn.RandomCrop(32, padding=4),
    trn.RandomHorizontalFlip(),
    Cutout(n_holes=1, length=16),
])

transform_test = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

# input = torch.rand(32, 3, 32, 32).cuda()
if args.model == "resnet18":
    net = models.resnet18()
elif args.model == "resnet50":
    net = models.resnet50()
# pdb.set_trace()
net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

if args.dataset == "cifar10":
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=10, bias=True)
    train_data = dset.CIFAR10('./cifar10', train=True, transform=transform_train, download=True)
    test_data = dset.CIFAR10('./cifar10', train=False, transform=transform_test)
elif args.dataset == "cifar100":
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=100, bias=True)
    train_data = dset.CIFAR100('./cifar100', train=True, transform=transform_train, download=True)
    test_data = dset.CIFAR100('./cifar100', train=False, transform=transform_test)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=128, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=128, shuffle=False
)

if torch.cuda.is_available():
    net.cuda()

# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# def adjust_lr(optimizer, epoch, lr_schedule=[60, 120, 180]): # POEM中使用的学习率调整方法
#     lr = args.lr
#     if epoch >= lr_schedule[0]:
#         lr *= 0.1
#     if epoch >= lr_schedule[1]:
#         lr *= 0.1
#     if epoch >= lr_schedule[2]:
#         lr *= 0.1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def train(epoch):
    # adjust_lr(optimizer, epoch)
    if (epoch + 1) % 10 == 0:
        args.lr *= 0.5
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    net.train()
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        x = net(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        loss.backward()
        optimizer.step()
    log.debug("loss:{0:5f}".format(loss))

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
            # log.debug("loss:{0:5f}".format(loss))
    accuracy = correct / len(test_loader.dataset)
    log.debug("accuracy:{0:5f}".format(accuracy))
    if accuracy > args.best_acc:
        args.best_acc = accuracy
        dir_save = args.model + "_"+ args.dataset + '_epoch' + str(epoch + 1) + '.pt'
        log.debug("save best checkpoints:epoch" + str(epoch + 1))
        torch.save(net.state_dict(), dir_save)

for epoch in range(350):
    # pdb.set_trace()
    log.debug("epoch" + str(epoch + 1) + ":")
    train(epoch)
    test(epoch)

# print(input.shape)
# output = net(input)
# print(output.shape)
# print(net.fc)
# print(net.fc.in_features, net.fc.out_features)
