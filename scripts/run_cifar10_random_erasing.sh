#!/usr/bin/env bash
python cifar.py  --dataset cifar10 --arch resnet --depth 20 --epochs 500 --train-batch 2000 --p 0.5 \
--resume "/media/zhenyue-qin/local/Research/Research-Yuriko/Yuriko-Folks/Folks-Random-Erasing/Random-Erasing/checkpoint/2019-12-23T12-19-33/model_best.pth.tar" \
-e
