sudo service docker start
sleep 1
docker build -t chai/mlrl:latest .
sudo service docker stop
