systemctl --user enable docker.service && systemctl --user start docker.service
sleep 1
docker run --gpus all -it --rm --name mlrl -v $(pwd):/tf/ --entrypoint=/bin/bash chai/mlrl:latest
