#!/usr/bin/env bash
python fashionmnist.py \
--dataset fashionmnist --arch resnet --depth 20 --epochs 500 --train-batch 2000 --drop 0.3