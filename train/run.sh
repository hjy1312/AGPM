#!/usr/bin/env sh

if [ ! -d "./log" ];then
   mkdir ./log
else
   :
fi
LOG=./log/log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/home/junyang/anaconda3/bin
nohup $PYDIR/python -B main.py --batchSize 200 \
 --cuda --ngpu 2 \
 --outf ./saved_model/model_`date +%Y-%m-%d-%H:%M:%S` 2>&1 | tee $LOG&
# --Dream_branch /home/junyang/experiment/Resnet34_progressive/stitching/saved_model/HP2F_gt_model_2019-07-31-09:35:35/model_29.pth \
