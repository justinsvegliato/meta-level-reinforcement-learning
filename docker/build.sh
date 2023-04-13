systemctl --user enable docker.service && systemctl --user start docker.service
sleep 1
docker build -t chai/mlrl:latest .