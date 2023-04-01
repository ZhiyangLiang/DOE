import torch
import numpy as np
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import pdb

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_test = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean, std),
])
test_data = dset.CIFAR10('../data/cifar10', train=False, transform=transform_test)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=128, shuffle=False
)

net = models.resnet18()
net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
net.fc = nn.Linear(in_features=net.fc.in_features, out_features=10, bias=True)
net.load_state_dict(torch.load('./ckpt/resnet18_cifar10_epoch147.pt'))
net.cuda()

def msp_get_ood_scores(loader):
    _score = []
    net.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            pdb.set_trace()
            output = net(data)
            # smax = to_np(output)
            smax = to_np(F.softmax(output, dim=1))
            _score.append(-np.max(smax, axis=1))
        return concat(_score).copy()

copy = msp_get_ood_scores(test_loader)
print(copy)
