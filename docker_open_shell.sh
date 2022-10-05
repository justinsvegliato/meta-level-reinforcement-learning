sudo service docker start
sleep 1
sudo docker run --runtime nvidia -it --rm --name mlrl -v $(pwd):/tf/ \
	-p :8888:8888 -p :6060:6060 -p :5678:5678 \
	--privileged=true \
	--entrypoint=/bin/bash \
	chai/mlrl:latest
sudo service docker stop