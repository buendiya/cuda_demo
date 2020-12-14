
# compile
nvcc --device-debug --debug name.cu -o build/name

# profile
sudo /usr/local/cuda/bin/nvprof ./add_cud
