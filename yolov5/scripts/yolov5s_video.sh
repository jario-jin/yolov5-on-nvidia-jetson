#!/bin/sh

model_name="aruco-yolov5s-v210819"
video_fn="/home/jario/Downloads/FullSizeRender.mov"

model_dir="weights"

if [ ! -d ""${model_dir}"" ];then
  echo "\033[31m[ERROR]: ${model_dir} not exist!: \033[0m"
  mkdir ${model_dir}
fi

if [ ! -f ""${model_dir}/${model_name}.pt"" ];then
  wget http://jario.ren/models/yolov5-5/${model_name}.pt -O ${model_dir}/${model_name}.pt
fi

python3 detect.py --source ${video_fn} --weights ${model_dir}/${model_name}.pt

