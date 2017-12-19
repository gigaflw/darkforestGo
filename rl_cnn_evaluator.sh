#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant 
# of patent rights can be found in the PATENTS file in the same directory.
# 

#!/bin/bash

NUM_GPU=1
OUTPUT_PATH=../dflog

echo num of gpu used = $NUM_GPU
echo output path = $OUTPUT_PATH

OUTPUT=$OUTPUT_PATH/cnn_eval
for i in `seq 1 $NUM_GPU`; do  
   echo "" > $OUTPUT-${i}.log
   th rl_network/rl_cnn_evaluator_resnet.lua --pipe_path $OUTPUT_PATH >> $OUTPUT-${i}.log 2>&1 &
   echo $!
done

# Wait until they are ready.
for i in `seq 1 $NUM_GPU`; do
    while true; do
      if grep -q "ready" $OUTPUT-${i}.log; then
        break
      fi
      sleep 1
    done
done

th rl_network/train_mcts.lua

