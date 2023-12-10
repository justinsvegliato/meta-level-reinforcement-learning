docker run --detach --rm -v $(pwd):/tf/ --workdir /tf --gpus all --name rlts-ablate-rewards chai/rlts:latest \
      python -m rlts.train.procgen_meta \
      --max_tree_size 256 --pretrained_percentile 0.1 --num_iterations 1000 \
      --collect_steps 4096 --n_collect_envs 64 --train_batch_size 64 --gpus 0 \
      --no_rewrite_rewards
docker logs -f rlts-ablate-rewards