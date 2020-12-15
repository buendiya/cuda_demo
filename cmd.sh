
# compile
nvcc --device-debug --debug tutorial1/name.cu -o build/name

# profile
sudo /usr/local/cuda/bin/nvprof ./add_cud
