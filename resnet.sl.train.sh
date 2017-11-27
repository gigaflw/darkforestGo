#!/usr/bin/env bash

th ./resnet/train.lua --batch_size 256 --data_augment --use_gpu --device 3 --n_res 10
