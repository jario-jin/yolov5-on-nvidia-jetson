#!/bin/sh

model_dir="models"
model_name="yolov5s"

if [ ! -d ""$model_dir"" ];then
  echo "\033[31m[ERROR]: $model_dir not exist!: \033[0m"
  mkdir $model_dir
fi

if [ ! -f ""$model_dir/${model_name}.pt"" ];then
  wget -O $model_dir/${model_name}.pt http://jario.ren/models/yolov5-5/${model_name}.pt
fi

cd yolov5
python3 generator.py ../$model_dir/${model_name}.pt

cd ..
mkdir build
mv $model_dir/${model_name}.wts build
cd build
cmake .. && make
sudo ./yolov5 -s ${model_name}.wts ${model_name}.engine s

