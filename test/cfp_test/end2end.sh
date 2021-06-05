#!/usr/bin/env sh
if [ ! -d "./log" ];then
   mkdir ./log
else
   :
fi
LOG=./log/log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=~/anaconda3/bin
nohup $PYDIR/python -B resnet34_cal_fea_progressive_end2end_train_together_all.py --batchSize 10 \
 --cuda --ngpu 1 \
 --Resnet34 /home/junyang/experiment/progressive_end2end/progressive_3_step/saved_model/valuable/model_2019-10-15-21:24:39/Resnet34_29.pth \
 --outf ./fea_resnet34 2>&1 | tee $LOG
