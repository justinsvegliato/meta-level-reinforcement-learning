# sudo service docker start
# sleep 1
# sudo docker run --runtime nvidia -it --rm --name mlrl_jupyter -v $(pwd):/tf/ \
# 	-p :8888:8888 -p :6060:6060 -p :5678:5678 \
# 	--privileged=true \
# 	chai/mlrl:latest
# sudo service docker stop

# docker run --gpus all -it --rm --name rlts_jupyter -v $(pwd):/tf/ -p :8989:8888 -p :5678:5678 --privileged=true chai/rlts:latest
docker run --gpus all -it --rm --name rlts_jupyter -v $(pwd):/tf/ -p :8989:8989 -p :5678:5678 --entrypoint /bin/bash --privileged=true chai/rlts:latest -c "jupyter server --port=8989 --no-browser --ip=0.0.0.0 --allow-root --ServerApp.token='' --ServerApp.password=''"
