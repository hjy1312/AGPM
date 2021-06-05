import numpy as np
import struct as st
import torch
import torch.nn as nn
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 45.0 - 1))
    return norm_angle

class Basic_Block(nn.Module):
    def __init__(self, feat_dim):
        super(Basic_Block, self).__init__()
        self.conv1 = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feat_dim)


    def forward(self, input, yaw):
        #input = input.unsqueeze(2).unsqueeze(3)
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        yaw = yaw.view(yaw.size(0),1,1,1)
        yaw = yaw.expand_as(x)

        feature = yaw * x + input
        feature = self.relu(feature)
        #feature = feature.squeeze()
        
        return feature

class Branch(nn.Module):
    def __init__(self, feat_dim):
        super(Branch, self).__init__()
        self.P2HP_Branch = Basic_Block(feat_dim)
        self.HP2HF_Branch = Basic_Block(feat_dim)
        self.HF2F_Branch = Basic_Block(feat_dim)

    def forward(self, input, yaw_HF, yaw_HP, yaw_P):
        input = input.unsqueeze(2).unsqueeze(3)
        x = self.P2HP_Branch(input,yaw_P)
        x = self.HP2HF_Branch(x,yaw_HP)
        x = self.HF2F_Branch(x,yaw_HF)
        #feature = x1
        feature = x.squeeze()
        
        return feature
