sudo service docker start
sleep 1
sudo docker run --runtime nvidia -it --rm --name mlrl -v $(pwd):/tf/mlrl \
	-p :8888:8888 -p :6060:6060 \
	--privileged=true \
	chai/mlrl:latest
sudo service docker stop
