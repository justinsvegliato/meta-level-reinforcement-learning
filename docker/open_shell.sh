systemctl --user enable docker.service && systemctl --user start docker.service
sleep 1
docker run -it --rm --name mlrl -v $(pwd):/tf/ \
	-p :8888:8888 -p :6060:6060 -p :5678:5678 \
	--entrypoint=/bin/bash \
	chai/mlrl:latest
