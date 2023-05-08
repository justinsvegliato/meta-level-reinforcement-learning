systemctl --user enable docker.service && systemctl --user start docker.service
sleep 1
docker build -t chai/mlrl:latest . --build-arg TF_SERVING_BUILD_OPTIONS="--copt=-mavx2 --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0"