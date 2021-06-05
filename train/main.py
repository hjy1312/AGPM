# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:50:42 2019

@author: msi
"""

from __future__ import print_function
import argparse
import os
import os.path as osp
import shutil
import pickle
import random
#from data_utils import get_train_test_data
import numpy as np
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat
import torch
import torch.nn as nn
from torch.nn import init
import pdb
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import sys
gpu_id = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
print('GPU: ',gpu_id)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     cudnn.deterministic = True
     #cudnn.benchmark = False
     #cudnn.enabled = False

setup_seed(2019)
       
from model_baseline import resnet34
from branch_util_512 import Branch
from dataset import ImageList
from torchvision.datasets import ImageFolder
from sklearn.metrics.pairwise import pairwise_distances
import math
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
#import pdb
from torch.autograd import Variable

def print_network(model,name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--out_class', type=int, default=6548, help='number of classes')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate, default=0.1')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD. default=0.9')
parser.add_argument('--gamma', type=float, default=1.01, help='')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay parameter. default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--model_save_step', type=int, default=100)
parser.add_argument('--Resnet34', default='', help="path to Resnet34 (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--training_list', default='/home/junyang/dataset/list/training_list_with_pose_with_sim_gt.txt', type=str, metavar='PATH',help='')
parser.add_argument('--manualSeed', type=int, default=2019, help='manual seed')
time1 = datetime.datetime.now()
time1_str = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
time1_str.replace(' ','_')

opt = parser.parse_args()
#opt.outf = opt.outf + str(split)
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass 

import glob
script_set = glob.glob('./*.py') + glob.glob('./*.sh')
for path in script_set:
    save_path = osp.join(opt.outf,osp.split(path)[-1])
    shutil.copyfile(path,save_path)


#cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
ngpu = int(opt.ngpu)

box = (16, 17, 214, 215)
transform=transforms.Compose([#transforms.Lambda(lambda x: x.crop(box)),
                             transforms.Resize((230,230)),
                             #transforms.Resize(opt.imageSize),                            
                             transforms.RandomGrayscale(p=0.1),
                             transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(),
                             transforms.RandomCrop((opt.imageSize,opt.imageSize)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])

tensor_dataset = ImageList(opt.training_list,transform)

def worker_init_fn(worker_id): 
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)
                         
dataloader = DataLoader(tensor_dataset,                        
                        batch_size=opt.batchSize,     
                        shuffle=True,     
                        num_workers=opt.workers,
                        worker_init_fn=worker_init_fn)  
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 :
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        #m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        #m.weight.data.normal_(1.0, 0.02)
        #m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant(m.bias.data, 0.0)
        #m.weight.data.normal_(0.0, 0.02)

def compute_accuracy(x, y):
     _, predicted = torch.max(x, dim=1)
     correct = (predicted == y).float()
     accuracy = torch.mean(correct) * 100.0
     return accuracy        

def zero_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 :
        init.constant(m.weight.data, 0.0)
        init.constant(m.bias.data, 0.0)

criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss(reduce=False)


Resnet34 = resnet34(num_classes = 62338, pretrained=False)


Resnet34.apply(weights_init)



if ngpu>1:
    Resnet34 = nn.DataParallel(Resnet34)


if opt.cuda:
    Resnet34.cuda()
    criterion.cuda()

optimizer = optim.SGD(Resnet34.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay = opt.weight_decay)

Resnet34.train()

cnt = 0
loss_log = []
print('initial learning rate is: {}'.format(opt.lr))
for epoch in range(opt.niter):
    if epoch == 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2.0
            print('lower learning rate to {}'.format(param_group['lr']))
    elif epoch == 10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/5.0
            print('lower learning rate to {}'.format(param_group['lr']))
    elif epoch == 15 or epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10.0
            print('lower learning rate to {}'.format(param_group['lr']))
    for i, (data,gt_data,label,pose1,pose2,pose3) in enumerate(dataloader,0):
        cnt += 1
        optimizer.zero_grad()
        batch_size = data.size(0)
        pose1 = pose1.float().unsqueeze(1)
        pose2 = pose2.float().unsqueeze(1)
        pose3 = pose3.float().unsqueeze(1)
        if opt.cuda:
            data = data.cuda()
            gt_data = gt_data.cuda()
            pose1 = pose1.cuda()
            pose2 = pose2.cuda()
            pose3 = pose3.cuda()
            label = label.cuda()
        inputv = Variable(data)
        gt_inputv = Variable(gt_data)
        label = Variable(label)
        pose1 = Variable(pose1)
        pose2 = Variable(pose2)
        pose3 = Variable(pose3)

        out,out_gt,fea = Resnet34(inputv,gt_inputv,pose1,pose2,pose3)
        t,_ = torch.max(torch.abs(out_gt),dim=1)
        w = out_gt / t.view(out_gt.size(0),1)
        loss1 = criterion(out,label)
        loss2 = (criterion2(fea,out_gt.detach()) * w.detach()).mean() * 2.0
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        loss_log.append([loss.item(),loss1.item(),loss2.item()])

        if (i+1)%opt.log_step == 0:
            accuracy = compute_accuracy(out,label).item()         
            print ('Epoch[{}/{}], Iter [{}/{}], training loss: {}, training loss1: {}, training loss2: {}, accuracy: {}%'.format(epoch+1,opt.niter,i+1, \
                  len(dataloader),loss.item(),loss1.item(),loss2.item(),accuracy))
        del data,inputv,label,gt_data,gt_inputv,out,out_gt,fea
    torch.save(Resnet34.state_dict(), '%s/Resnet34_%d.pth' % (opt.outf, epoch))           
        
loss_log = np.array(loss_log)
plt.plot(loss_log[:,0], label="Training Loss")
plt.plot(loss_log[:,1], label="Training Loss1")
plt.plot(loss_log[:,2], label="Training Loss2")
plt.legend(loc='upper right')
plt.xlabel("Iter/10")
plt.ylabel("Loss")
plt.show()
filename = os.path.join(opt.outf, ('Loss_log_'+time1_str+'.png'))
plt.savefig(filename, bbox_inches='tight')























