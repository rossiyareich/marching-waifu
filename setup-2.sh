#! /bin/bash

# Setup conda env
eval "$(conda shell.bash hook)"
conda create -n marching-waifu python=3.10
conda activate marching-waifu
pip install --upgrade pip

# Install Jupyter, Pandas, Numpy, gdown, Pillow, IPython
pip install jupyter pandas numpy Pillow ipython gdown

# Install TensorFlow, Keras
pip install tensorflow==2.12.*
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install keras

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia