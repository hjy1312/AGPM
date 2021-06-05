#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path
import torch.utils.data as data

def default_loader(path):
    img = Image.open(path)
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            index, imgPath = line.strip().rstrip('\n').split(' ')
            imgList.append((imgPath))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgPath = self.imgList[index]
        img = self.loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgList)
