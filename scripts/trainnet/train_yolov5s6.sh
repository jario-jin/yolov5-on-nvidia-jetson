#!/bin/bash
PYTHON=${PYTHON:-"python3"}
cd yolov5


standard_model="yolov5s"
img_size=640
batch_size=8
n_epochs=32

dataset_name="aruco_marker_v2108"
pre_trained_weights="weights/yolov5s.pt"

pre_trained_sign="cocobased"
hyp_yaml="data/hyp.aruco.yolov5s6.yaml"


$PYTHON train.py --img-size ${img_size} --batch-size ${batch_size} --epochs ${n_epochs} --data ../data/${dataset_name}.yaml --cfg ./models/${standard_model}.yaml --weights ${pre_trained_weights} --hyp ${hyp_yaml} #--evolve 


# --exp-name ${standard_model}_${pre_trained_sign}_${dataset_name}_b${batch_size}_i${img_size}_e${n_epochs}

echo 'Done!'

