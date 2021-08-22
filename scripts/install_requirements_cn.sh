echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
echo 'export PATH=/home/nvidia/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64' >> ~/.bashrc
source ~/.bashrc

pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

sudo apt-get install curl
sudo apt-get install wget
sudo apt-get install python3-matplotlib
sudo apt-get install python3-pip libopenblas-dev libopenmpi-dev libomp-dev
sudo -H pip3 install future
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install Cython

pip3 install -r requirements.txt
pip3 install seaborn --no-deps
pip3 install wedet
pip3 install pycocotools

wget http://jario.ren/whl/jetpack44-45/torch-1.9.0-cp36-cp36m-linux_aarch64.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl
pip3 install numpy torch-1.9.0-cp36-cp36m-linux_aarch64.whl
rm torch-1.9.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
wget http://jario.ren/whl/torchvision-v0.10.0.zip -O torchvision-v0.10.0.zip
unzip torchvision-v0.10.0.zip
cd torchvision-v0.10.0
export BUILD_VERSION=0.10.0  # where 0.10.0 is the torchvision version  
python3 setup.py install --user
cd ../
rm torchvision-v0.10.0.zip
sudo rm -r torchvision-v0.10.0

sudo pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda

bash scripts/manage_swap.sh
