#! /bin/bash

# Install miniconda
sudo apt update
sudo apt upgrade
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Setup 3rd party packages
sudo apt install build-essential
sudo apt install nvidia-cuda-toolkit
sudo apt install unzip

# Reboot
echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
sudo reboot now

# Setup conda env
conda create -n marching-waifu python=3.10
conda activate marching-waifu
pip install --upgrade pip

# Install Jupyter, Pandas, Numpy, gdown, Pillow, IPython
conda install jupyter pandas numpy Pillow ipython
conda install -c conda-forge gdown

# Install CUDAToolkit, CUDNN
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Install NVCC
conda install -c nvidia cuda-nvcc=11.3.58
# Configure the XLA cuda directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Copy libdevice file to the required path
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/

# Install TensorFlow, Keras
pip install tensorflow==2.12.*
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
conda install -c conda-forge keras

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia