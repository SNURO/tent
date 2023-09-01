"""
Network architectures and functionality for adding stats layers to networks // originated from BUFR
"""

from __future__ import division, print_function, absolute_import
import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    """
    Cifar sizes, printing x.shape and every out.shape in the forward pass
    torch.Size([128, 3, 32, 32])    - Input                                                                                                                                                                           
    torch.Size([128, 64, 32, 32])   - First layers (conv+bn) output                                                                                                                                                                             
    torch.Size([128, 64, 32, 32])   - self.layer1 output                                                                                                                                                                            
    torch.Size([128, 128, 16, 16])  - self.layer2 output                                                                                                                                                                                
    torch.Size([128, 256, 8, 8])    - self.layer3 output                                                                                                                                                                                
    torch.Size([128, 512, 4, 4])    - self.layer4 output                                                                                                                                                                                
    torch.Size([128, 512, 1, 1])    - avg_pool output                                                                                                                                                                            
    torch.Size([128, 512])          - reshape output                                                                                                                                                                            
    torch.Size([128, 10])           - linear output
    
    Camelyon using this resnet:
    torch.Size([128, 3, 96, 96])
    torch.Size([128, 64, 96, 96])
    torch.Size([128, 64, 96, 96])
    torch.Size([128, 128, 48, 48])
    torch.Size([128, 256, 24, 24])
    torch.Size([128, 512, 12, 12])
    torch.Size([128, 512, 3, 3])
    torch.Size([128, 4608])   <- breaks here as this is should be 128, 512 after reshape then go to 
    
    If we use the resnet in nets_wilds:
    torch.Size([128, 3, 96, 96])   - Input                                                                                                                                                                             
    torch.Size([128, 64, 48, 48])  - First layers (conv+bn) output                                                                                                                                                                             
    torch.Size([128, 64, 24, 24])  - Max pool output                                                                                                                                                                             
    torch.Size([128, 64, 24, 24])  - self.layer1 output                                                                                                                                                                              
    torch.Size([128, 128, 12, 12]) - self.layer2 output                                                                                                                                                                              
    torch.Size([128, 256, 6, 6])   - self.layer3 output                                                                                                                                                                              
    torch.Size([128, 512, 3, 3])   - self.layer4 output                                                                                                                                                                               
    torch.Size([128, 512, 1, 1])   - AdaptiveAvgPool2d output                                                                                                                                                                           
    torch.Size([128, 512])         - reshape output                                                                                                                                                                             
    torch.Size([128, 1])           - linear output

    """


def ResNet18(n_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes)


def ResNet34(n_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)

#50layer가 아닌뎅? Bottleneck 이라 그런듯. depth가 깊어지면 basicblock 대신 bottleneck 사용
def ResNet50(n_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes)


def ResNet101(n_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=n_classes)
