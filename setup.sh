#! /bin/bash

# -----------------------------------------------------
# Reboot after install!
# -----------------------------------------------------
# Make sure to copy over the following files to ~/
# cudnn-local-repo-ubuntu2204-8.9.1.23_1.0-1_amd64.deb
# ----------------------------------------------------- 

# Update submodules
git submodule update --init --recursive

# Setup prerequisites
sudo apt update
sudo apt install build-essential cmake unzip

# Setup latest NVIDIA drivers
sudo apt remove --purge nvidia-*
sudo apt autoremove --purge
sudo apt install nvidia-driver-525

# Setup cuda-toolkit 11.8
cd ~
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
sudo bash -c “echo /usr/local/cuda/lib64 > /etc/ld.so.conf”
sudo ldconfig
cd ~
rm -rf cuda_11.8.0_520.61.05_linux.run

# Setup cuDNN 8.9.1
mkdir cudnn_install
mv cudnn-local-repo-ubuntu2204-8.9.1.23_1.0-1_amd64.deb cudnn_install
cd cudnn_install
ar -x cudnn-local-repo-ubuntu2204-8.9.1.23_1.0-1_amd64.deb
tar -xvf data.tar.xz
cd var/cudnn-local-repo-ubuntu2204-8.9.1.23/
sudo dpkg -i libcudnn8_8.9.1.23-1+cuda11.8_amd64.deb
sudo dpkg -i libcudnn8-dev_8.9.1.23-1+cuda11.8_amd64.deb
sudo dpkg -i libcudnn8-samples_8.9.1.23-1+cuda11.8_amd64.deb
cd ~
rm -rf cudnn_install

# Setup anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
sh Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.bashrc
cd ~
rm -rf Anaconda3-2023.03-1-Linux-x86_64.sh

# Setup conda env
eval "$(conda shell.bash hook)"
conda create -n marching-waifu python=3.10
conda activate marching-waifu
pip install --upgrade pip

# Install gdown
pip install gdown

# Install TensorFlow, Keras
pip install tensorflow==2.12.*
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install keras

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Download pretrained DeepDanbooru weights
mkdir ext/weights/
cd ext/weights/
wget https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip
unzip deepdanbooru-v3-20211112-sgd-e28.zip -d ../AnimeFaceNotebooks/deepdanbooru_model/
cd ../
rm -rf weights
cd ../

: '
# Build instant-ngp
# ... (TODO)
    # NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh

    # Setup Optix 7.7
    sudo chmod +x NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh
    sh NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh
    echo "OptiX_INSTALL_DIR=/home/vcg0/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64" >> ~/.bashrc
    source ~/.bashrc
    cd ~
    rm -rf NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh

    # Setup Vulkan SDK
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
    sudo apt install vulkan-sdk
'