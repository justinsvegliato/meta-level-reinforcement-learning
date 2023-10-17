systemctl --user enable docker.service && systemctl --user start docker.service
sleep 1
docker run --gpus all -it --rm --name rlts -v $(pwd):/tf/ --entrypoint=/bin/bash chai/rlts:latest
# docker run --runtime nvidia -it --rm --name mlrl_shell -v $(pwd):/tf/ --entrypoint=/bin/bash chai/mlrl:latest
