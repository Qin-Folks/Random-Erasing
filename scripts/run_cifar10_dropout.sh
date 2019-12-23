#!/usr/bin/env bash
python cifar.py \
--dataset cifar10 --arch resnet --depth 20 --epochs 500 --train-batch 2000 --dr