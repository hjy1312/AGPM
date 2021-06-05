# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:44:11 2019

@author: msi
"""

import os, sys, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bisect
import pickle
import math
import random as rd
import struct as st
from scipy import spatial
from sklearn import metrics
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity


# {{{ draw roc curve
def draw_roc(fpr, tpr, title, img_name="roc.png"):
    plt.figure(figsize=(16, 8))
    plt.yticks(np.arange(0.0, 1.05, 0.05))
    plt.xticks(np.arange(0.0, 1.05, 0.05))
    plt.title(title)
    plt.plot(fpr, tpr, linewidth=1, color='r')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.savefig(img_name)

def draw_multiple_roc(tprs, fprs, labels, title, img_name):
    plt.figure(figsize=(16,8))
    plt.yticks(np.arange(0.0, 1.05, 0.05))
    plt.xticks(np.arange(0.0, 1.05, 0.05))
    for i in range(len(tprs)):
        plt.plot(fprs[i], tprs[i], label=labels[i])
    plt.legend(loc=4)
    plt.title(title)
    plt.savefig(img_name)
# }}}

def load_feat(feat_file):
    feats = list()
    with open(feat_file, 'rb') as in_f:
        feat_num, feat_dim = st.unpack('ii', in_f.read(8))
        for i in range(feat_num):
            feat = np.array(st.unpack('f'*feat_dim, in_f.read(4*feat_dim)))
            feats.append(feat)
    return feats


#calculate eer{{{
def calc_eer(fpr, tpr, method=0):
    if method == 0:
        min_dis, eer = 100.0, 1.0
        for i in range(fpr.size):
            if(fpr[i]+tpr[i] > 1.0):
                break
            mid_res = abs(fpr[i]+tpr[i]-1.0)
            if(mid_res < min_dis):
                min_dis = mid_res
                eer = fpr[i]
        return eer
    else:
        f = lambda x: np.interp(x, fpr, tpr)+x-1
        return fsolve(f, 0.0)
 #}}}

def eval_roc(protocol_dir, pair_type, split_name, frontal_feats, profile_feats):
    labels, scores = [],[]
    for idx, pair_file in enumerate(['diff.txt', 'same.txt']):
        label = idx
        full_pair_file = protocol_dir+'/'+split_name+'/'+pair_file
        with open(full_pair_file, 'r') as in_f:
            for line in in_f:
                record = line.strip().split(',')
                pair1, pair2 = int(record[0]),int(record[1])
                if pair_type == 'FF':
                    vec1, vec2 = frontal_feats[pair1-1], frontal_feats[pair2-1]
                else:
                    vec1, vec2 = frontal_feats[pair1-1], profile_feats[pair2-1]
                score = 1-spatial.distance.cosine(vec1,vec2)
                scores.append(score)
                labels.append(label)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    tpr001 = tpr[ bisect.bisect_left(fpr,0.001) ]
    tpr01 = tpr[ bisect.bisect_left(fpr,0.01) ]
    #draw_roc(fpr,tpr,title='center loss')
    auc = metrics.auc(fpr, tpr)
    eer = calc_eer(fpr, tpr)
    return auc, eer, tpr001, tpr01
    

def eval_roc_main(frontal_feats,profile_feats):
    protocol_dir = './cplfw_protocol/Protocol/Split'
    pair_types = [ 'FP']
    split_num = 10

    for pair_type in pair_types:
        print_info = 'Frontal-Frontal' if pair_type=='FF' else 'Frontal-Profile'
        print('----- result for {} -----'.format(print_info))
        aucs, eers, tprs001, tprs01 = list(), list(), list(), list()
        for split_id in range(split_num):
            split_name = str(split_id+1)
            if len(split_name)<2: split_name = '0'+split_name
            auc, eer, tpr001, tpr01 = eval_roc(protocol_dir, pair_type, split_name, frontal_feats, profile_feats)
            #print('{} split {}, auc: {}, eer: {}'.format(pair_type,split_name,auc,eer))
            aucs.append(auc)
            eers.append(eer)
            tprs001.append(tpr001)
            tprs01.append(tpr01)
        avg_auc = 100 * np.mean(aucs)
        avg_eer = 100 * np.mean(eers)
        std_auc = 100 * np.std(aucs)
        std_eer = 100 * np.std(eers)
        avg_tpr001 = 100 * np.mean(tprs001)
        avg_tpr01 = 100 * np.mean(tprs01)
        print('Average auc: {}%'.format(100 * np.mean(aucs)))
        print('Average eer: {}%'.format(100 * np.mean(eers)))
        print('Average TPR@FPR=0.001: {}%'.format(100 * np.mean(tprs001)))
        print('Average TPR@FPR=0.01: {}%'.format(100 * np.mean(tprs01)))
    return avg_auc,avg_eer,avg_tpr001,avg_tpr01,std_auc,std_eer

if __name__ == '__main__':
    _,_,_,_ = eval_roc_main()
