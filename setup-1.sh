#! /bin/bash

# Setup NVIDIA drivers with CUDA 11.8 toolkit
sudo apt update
sudo apt install build-essential
sudo apt install nvidia-driver-525
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
echo "export PATH='/usr/local/cuda/bin:$PATH'" >> ~/.bashrc
echo "export LD_LIBRARY_PATH='/usr/local/cuda/lib64:$LD_LIBRARY_PATH'" >> ~/.bashrc
source ~/.bashrc
sudo bash -c “echo /usr/local/cuda/lib64 > /etc/ld.so.conf”
sudo ldconfig
ldconfig -p | grep cuda
cat /usr/local/cuda/version.json | grep version -B 2
rm -rf ~/cuda_11.8.0_520.61.05_linux.run

# Setup cuDNN 8.9.0
wget -O cudnn.deb https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.0/local_installers/11.8/cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb/
mkdir cudnn_install
mv cudnn.deb cudnn_install
cd cudnn_install
ar -x cudnn.deb
tar -xvf data.tar.xz
cd var/cudnn-local-repo-ubuntu2204-8.8.0.121/
sudo dpkg -i libcudnn8_8.8.0.121-1+cuda11.8_amd64.deb
sudo dpkg -i libcudnn8-dev_8.8.0.121-1+cuda11.8_amd64.deb
sudo dpkg -i libcudnn8-samples_8.8.0.121-1+cuda11.8_amd64.deb
cat /usr/include/x86_64-linux-gnu/cudnn_version_v8.h | grep CUDNN_MAJOR -A 2
cd ~
rm -r cudnn_install

# Install miniconda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
rm ~/Miniconda3-latest-Linux-x86_64.sh

# Setup 3rd party packages
sudo apt install unzip