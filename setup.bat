:: -----------------------------------------------------
:: Make sure anaconda is installed and 
:: conda env is up! (python=3.10)
:: -----------------------------------------------------
:: Make sure to have the following installed:
::  VS2022 build tools + CMake
::  Latest NVIDIA drivers
::  cuda-toolkit 11.8
::  cuDNN 8.9.1
:: -----------------------------------------------------

:: Update submodules
git submodule update --init --recursive

:: Install 3rd party packages
python -m pip install --upgrade pip
pip install gdown ipykernel ipywidgets

:: Install Tensorflow, Keras
pip install tensorflow-directml-plugin
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install keras

:: Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Download pretrained DeepDanbooru weights
mkdir "ext/weights/"
cd "ext/weights/"
powershell -Command "& Invoke-WebRequest https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip -OutFile deepdanbooru-v3-20211112-sgd-e28.zip"
powershell -Command "& Expand-Archive -Force deepdanbooru-v3-20211112-sgd-e28.zip ../AnimeFaceNotebooks/deepdanbooru_model"
cd ..
powershell -Command "& rm -Recurse weights"
cd ..

:: Build instant-ngp
:: ... (TODO)
    ::  Optix 7.7
    ::  Vulkan SDK