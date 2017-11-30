#!/usr/bin/env bash

th ./resnet/train.lua --log_file resnet.ckpt/log.txt --batch_size 256 --n_res 10 --data_augment --use_gpu --device 3 --epoch_per_test 50 --epoch_per_ckpt 100 --epochs 1000 --log_file 'resnet.ckpt/log.txt'
