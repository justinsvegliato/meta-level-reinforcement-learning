#!/bin/bash

for i in 0 3 4
do
   # echo $i
   # srun bash -c "systemctl --user enable docker.service && systemctl --user start docker.service;sleep 1;docker run --gpus all --name rlts$1 chai/rlts:latest echo "hello from container $1 on " `hostname`"
   # srun bash -c "./run_meta_training.sh $i"
   docker run --detach --rm -v $(pwd):/tf/ --workdir /tf --gpus all --name rlts-ablate-state chai/rlts:latest python -m rlts.train.procgen_meta \
      --max_tree_size 256 --pretrained_percentile 0.1 --num_iterations 1000 --collect_steps 4096 --n_collect_envs 64 --train_batch_size 64 --gpus 0 --ablate_state_tokenise
done
