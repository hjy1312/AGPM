#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path
import math
import torch.utils.data as data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle,threshold):
    norm_angle = sigmoid(10 * (abs(angle) / threshold - 1))
    return norm_angle

def default_loader(path):
    img = Image.open(path)
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            index, imgPath, yaw = line.strip().rstrip('\n').split(' ')
            yaw = float(yaw)
            pose1 = norm_angle(yaw+5,25)
            pose2 = norm_angle(yaw-15,25)
            pose3 = norm_angle(yaw-35,25)       
            imgList.append((imgPath,pose1,pose2,pose3))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgPath,pose1,pose2,pose3 = self.imgList[index]
        img = self.loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)

        return img,pose1,pose2,pose3

    def __len__(self):
        return len(self.imgList)
