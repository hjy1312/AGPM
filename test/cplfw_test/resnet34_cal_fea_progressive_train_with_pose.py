# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 19:50:04 2019

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
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pdb
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('../../train')
from model_test import resnet34
from dataset_cplfw import ImageList
#from net_ArcFace import ArcFace
from eval_auc_eer_tpr_ import eval_roc_main
import math
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
#import pdb
from torch.autograd import Variable

def cal_cosine_distance(a,b):
    dot_product = np.sum(a*b)
    norm_a = math.sqrt(np.sum(np.square(a)))
    norm_b = math.sqrt(np.sum(np.square(b)))
    cosine_similarity = float(dot_product) / (norm_a * norm_b)
    return cosine_similarity

def print_network(model,name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))

def get_threshold(split,scores,labels):
    max_acc = 0.0
    best_threshold = 0.0
    split_set = list(range(1,11))
    split_set.pop(split-1) #split-1 is the index
    th_list = np.arange(-1,1.001,0.001)
    for th_ind in range(th_list.shape[0]):
        sum_predict_right = 0.0
        num = 0.0
        threshold = th_list[th_ind]
        for i in split_set:
            score = scores[i-1]
            label = labels[i-1]
            predicted = score>threshold
            predict_right = (predicted==label)
            sum_predict_right = sum_predict_right + float(predict_right.sum())
            num = num + float(label.shape[0])
        acc = sum_predict_right / num
        if acc>max_acc:
            max_acc = acc
            best_threshold = threshold
    return best_threshold

def cal_acc(score,label,threshold):
    predicted = score>threshold
    predict_right = (predicted==label)
    acc = float(predict_right.sum()) / float(label.shape[0])
    return acc

def cal_acc_model(model_num=30,scores_F_set=None,labels_F_set=None,scores_P_set=None,labels_P_set=None):
    acc_set = []
    std_set = []
    for model_cnt in range(1,model_num+1):
        acc_F = []
        acc_P = []
        scores_F = scores_F_set[model_cnt - 1]
        labels_F = labels_F_set[model_cnt - 1]
        scores_P = scores_P_set[model_cnt - 1]
        labels_P = labels_P_set[model_cnt - 1]       
        for split in range(1,11):
            score = scores_F[split-1]
            label = labels_F[split-1]
            threshold = get_threshold(split,scores_F,labels_F)
            acc_F.append(cal_acc(score,label,threshold))
            #print(acc_F)
            
            score = scores_P[split-1]
            label = labels_P[split-1]
            threshold = get_threshold(split,scores_P,labels_P)
            acc_P.append(cal_acc(score,label,threshold))
        #acc_F /= 10.0
        #acc_P /= 10.0
        acc_F = np.array(acc_F)
        acc_P = np.array(acc_P)
        mean_acc_F = np.mean(acc_F)
        mean_acc_P = np.mean(acc_P)
        std_acc_F = np.std(acc_F) * 100.0
        std_acc_P = np.std(acc_P) * 100.0
        acc_set.append([mean_acc_F,mean_acc_P])
        std_set.append([std_acc_F,std_acc_P])
    acc_set = np.array(acc_set)
    std_set = np.array(std_set)
    return acc_set,std_set

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--out_class', type=int, default=6548, help='number of classes')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--model_save_step', type=int, default=100)
parser.add_argument('--Resnet34', default='', help="path to resnet34")
parser.add_argument('--Deform_net', default='', help="path to Deform_net")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--dim_features', type=int, default=512, help='dim of features to use')
parser.add_argument('--gallery_list', default='./aligned_cplfw_list_with_pose_2.txt', type=str, metavar='PATH',
                    help='path to gallery list (default: none)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
time1 = datetime.datetime.now()
time1_str = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
time1_str.replace(' ','_')
gpu_id = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
print('GPU: ',gpu_id)
opt = parser.parse_args()
print(opt)
try:
    os.makedirs(opt.outf)
except OSError:
    pass 

model_index_set = range(25,30)
path_prefix = osp.split(opt.Resnet34)[0]
aucs, eers, tprs001, tprs01, std_aucs, std_eers = list(), list(), list(), list(), list(), list()
scores_F_set,labels_F_set,scores_P_set,labels_P_set  = list(), list(), list(), list()
for model_ii in model_index_set:
    path_prefix = osp.split(opt.Resnet34)[0]
    model_name = 'Resnet34_' + str(model_ii) + '.pth'
    opt.Resnet34 = osp.join(path_prefix,model_name)
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    
    cudnn.benchmark = True
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    ngpu = int(opt.ngpu)
    
    pre_model = torch.load(opt.Resnet34)
    pretrained_model_2_gpu = 'module' in list(pre_model.keys())[0]

    #Resnet34 = resnet34(num_classes = 93436)
    Resnet34 = resnet34(num_classes = 62338)
    
    if opt.Resnet34 != '':
        if pretrained_model_2_gpu:
            Resnet34 = nn.DataParallel(Resnet34)
        Resnet34.load_state_dict(pre_model,strict=True)
        if pretrained_model_2_gpu:
            Resnet34 = Resnet34.module
    for param in Resnet34.parameters():
        param.requires_grad = False
    
    if ngpu>1:
        Resnet34 = nn.DataParallel(Resnet34)
    
    if opt.cuda:
        Resnet34.cuda()
    
    Resnet34.eval()
    F_fea = []
    P_fea = []
    
    for list_name in ['./aligned_cplfw_list_with_pose_2.txt']:
    #for list_name in ['./aligned_Pair_list_P.txt']:
        opt.gallery_list = list_name
        F_True = 1
        
        box = (16, 17, 214, 215)
        transform=transforms.Compose([transforms.Lambda(lambda x: x.crop(box)),
                                     transforms.Resize((230,230)),
                                     transforms.CenterCrop((opt.imageSize,opt.imageSize)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
        tensor_dataset_gallery = ImageList(opt.gallery_list,transform)
                            
        dataloader_gallery = DataLoader(tensor_dataset_gallery,                          
                                        batch_size=opt.batchSize,    
                                        shuffle=False,     
                                        num_workers=opt.workers)   
        
        
        
        cnt = 0
        
        for i, (real_cpu,w1,w2,w3) in enumerate(dataloader_gallery):
            print('processing %d th batch' %(i+1))
            #print(data)
            batch_size = real_cpu.size(0)
            w1 = w1.float()
            w2 = w2.float()
            w3 = w3.float()
            if opt.cuda:
                w1 = w1.cuda()
                w2 = w2.cuda()
                w3 = w3.cuda()
                real_cpu = real_cpu.cuda()
            w1 = Variable(w1)
            w2 = Variable(w2)
            w3 = Variable(w3)
            inputv = Variable(real_cpu)
            feature = Resnet34(inputv,w1,w2,w3)
            feature = feature.cpu().data.numpy()
            for j in range(batch_size):
                if  F_True:
                    F_fea.append(feature[j,:].reshape(feature.shape[1]))                    
                else:
                    P_fea.append(feature[j,:].reshape(feature.shape[1]))
                cnt += 1
        print('finished')
    #exit(0)
    P_fea = F_fea
    print(model_ii)
    auc, eer, tpr001, tpr01, std_auc, std_eer = eval_roc_main(F_fea,P_fea)
    std_aucs.append(std_auc)
    std_eers.append(std_eer)
    aucs.append(auc)
    eers.append(eer)
    tprs001.append(tpr001)
    tprs01.append(tpr01)
    
    root_dir = './cplfw_protocol/Protocol/Split/'
    scores_F = []
    labels_F = []
    scores_P = []
    labels_P = []
    for split in ['01','02','03','04','05','06','07','08','09','10']:
        #save_subdir = osp.join(save_dir,'verification_resnet34_split'+str(int(split)))
        #if not osp.exists(save_subdir):
            #os.makedirs(save_subdir)
        
        for s in ['FF']:
            cos_dis_set = []
            mated_label_set = []
            src_dir0 = osp.join(root_dir)
            src_dir = osp.join(src_dir0,split)
            if s=='FF':
                fea_set1 = F_fea
                fea_set2 = F_fea
            else:
                fea_set1 = F_fea
                fea_set2 = P_fea
                
            cnt = 0
            f = open(osp.join(src_dir,'same.txt'),'r')
            for line in f.readlines():
                cnt += 1
                #print('match: %s, split: %s, index: %d' %(s,split,cnt))
                ind1,ind2 = line.strip().rstrip('\n').split(',')
                ind1,ind2 = int(ind1),int(ind2)
                fea1 = fea_set1[ind1-1]
                fea2 = fea_set2[ind2-1]
                cos_dis = cal_cosine_distance(fea1,fea2)
                cos_dis_set.append(cos_dis)
                mated_label_set.append(1)
            f.close()
            f = open(osp.join(src_dir,'diff.txt'),'r')
            for line in f.readlines():
                cnt += 1
                #print('match: %s, split: %s, index: %d' %(s,split,cnt))
                ind1,ind2 = line.strip().rstrip('\n').split(',')
                ind1,ind2 = int(ind1),int(ind2)
                fea1 = fea_set1[ind1-1]
                fea2 = fea_set2[ind2-1]
                cos_dis = cal_cosine_distance(fea1,fea2)
                cos_dis_set.append(cos_dis)
                mated_label_set.append(0)
            f.close()
            cos_dis_set = np.array(cos_dis_set)
            mated_label_set = np.array(mated_label_set)
            if s == 'FF':
                scores_F.append(cos_dis_set)
                labels_F.append(mated_label_set)
                scores_P.append(cos_dis_set)
                labels_P.append(mated_label_set)
    scores_F_set.append(scores_F)
    labels_F_set.append(labels_F)
    scores_P_set.append(scores_P)
    labels_P_set.append(labels_P)
save_dir = osp.join(path_prefix,'auc_eer_tpr_cplfw')
if not osp.exists(save_dir):
    os.makedirs(save_dir)
src_path = './cplfw_show_std.py'
dst_path = osp.join(save_dir,'cplfw_show_std.py')
shutil.copyfile(src_path,dst_path)
src_path = './show_cplfw.py'
dst_path = osp.join(save_dir,'show_cplfw.py')
shutil.copyfile(src_path,dst_path)
savemat(osp.join(save_dir,'auc.mat'),{'auc':np.array(aucs)})
savemat(osp.join(save_dir,'eer.mat'),{'eer':np.array(eers)})
savemat(osp.join(save_dir,'std_auc.mat'),{'std_auc':np.array(std_aucs)})
savemat(osp.join(save_dir,'std_eer.mat'),{'std_eer':np.array(std_eers)})
savemat(osp.join(save_dir,'tpr001.mat'),{'tpr001':np.array(tprs001)})
savemat(osp.join(save_dir,'tpr01.mat'),{'tpr01':np.array(tprs01)})

acc_set,std_set = cal_acc_model(len(model_index_set),scores_F_set,labels_F_set,scores_P_set,labels_P_set)
save_txt = osp.join(path_prefix,'acc_cplfw_2.txt')
np.savetxt(save_txt,acc_set,fmt=['%s']*acc_set.shape[1],newline = '\n')
save_txt = osp.join(path_prefix,'std_cplfw.txt')
np.savetxt(save_txt,std_set,fmt=['%s']*std_set.shape[1],newline = '\n')
