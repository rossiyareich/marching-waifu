:: Make sure conda installed and conda init is ran
:: Make sure to have CUDA Toolkit installed

:: Setup conda env
conda create -n marching-waifu python=3.10
conda activate marching-waifu
pip install --upgrade pip

:: Install Jupyter, Pandas, Numpy, gdown, Pillow, IPython
conda install jupyter pandas numpy gdown Pillow ipython

:: Install Tensorflow, Keras
pip install tensorflow-directml-plugin
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
conda install -c conda-forge keras

:: Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia