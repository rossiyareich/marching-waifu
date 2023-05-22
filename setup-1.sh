#! /bin/bash

# Install miniconda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc

# Setup 3rd party packages
sudo apt update
sudo apt install build-essential
sudo apt install nvidia-cuda-toolkit
sudo apt install unzip

# Reboot
sudo reboot now