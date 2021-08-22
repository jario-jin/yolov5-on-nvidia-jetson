# Yolov5 TensorRT Conversion & Deployment on Jetson Nano & TX2 & Xavier [Ultralytics EXPORT]
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://gitee.com/jario-jin/yolov5-on-nvidia-jetson/blob/master/LICENSE)
[![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)]()

[<img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash-export-competition.png">](https://github.com/ultralytics/yolov5/discussions/3213)


# Notes
This repository is to deploy Yolov5 to NVIDIA Jetson platform and accelerate it through TensorRT technology.
Available devices such as Nano (472 GFLOPS), TX2 (1.33 TFLOPS), Xavier NX (21 TOPS) and AGX Xavier (32TOPS).

# JetPack
Firstly you should get latest JetPack *v4.5.1* from NVIDIA and boot it onto Jetson Nano. 
You could find JetPack download options ***[here](https://developer.nvidia.com/embedded/jetpack)***


# Conversion Steps
1. Clone this repository and Enter it:
   ```shell
   git clone https://gitee.com/jario-jin/yolov5-on-nvidia-jetson.git
   cd yolov5-on-nvidia-jetson
   ```
   
2. Install requirements for JetPack:
   ```shell
   bash scripts/install_requirements.sh
   ```
   
3. Convert the model:
   ```shell
   bash scripts/convert_models.sh
   ```
   
4. Test detection speed and accuracy:
   ```shell
   wget http://images.cocodataset.org/zips/val2017.zip
   # Download instances_val2017.json
   gdown https://drive.google.com/uc?id=13x1Nc7Jpjz-dDtrAP2A8Jf2LXWscV0q6
   unzip val2017.zip
   # Inference with Python front-end
   python3 inference.py --input "val2017/*.jpg" --gt instances_val2017.json
   # Inference with C++ front-end
   bash scripts/cpp_tensorrt_coco_eval.sh
   ```

5. Test with a webcam (id=0):
   ```shell
   python3 inference.py --webcam
   ```

# Accuracy and speed test results on COCO val

|Device    |size   |FP16/INT8 |Front-end |mAP<sup>val<br>0.5:0.95 |mAP<sup>small<br> |Avg.Infer.Time |
| ------   |:---:  |:---:     |:---:     |:---:                   |:---:             |:---:          |
|Nano      |640    |FP16      |Python    |33.3                    |17.3              |105.8 ms (+- 13.5 ms) |
|Nano      |640    |FP16      |C++       |33.7                    |17.6              |87.0 ms (+- 4.2 ms) |
|Nano      |320    |FP16      |Python    |28.1                    |7.7               |38.1 ms (+- 3.3 ms) |
|Nano      |320    |FP16      |C++       |28.1                    |7.3               |32.9 ms (+- 4.1 ms) |
|Xavier NX |640    |FP16      |Python    |33.3                    |17.3              |37.5 ms (+- 4.5 ms) |
|Xavier NX |640    |FP16      |C++       |33.7                    |17.6              |25.5 ms (+- 2.3 ms) |
|Xavier NX |320    |FP16      |Python    |28.2                    |7.8               |26.2 ms (+- 1.8 ms) |
|Xavier NX |320    |FP16      |C++       |28.1                    |7.1               |15.1 ms (+- 1.7 ms) |
|AGX Xavier|640    |FP16      |Python    |33.3                    |17.3              |30.2 ms (+- 3.3 ms) |
|AGX Xavier|640    |FP16      |C++       |33.7                    |17.6              |18.5 ms (+- 2.1 ms) |
|AGX Xavier|320    |FP16      |Python    |28.1                    |7.8               |23.5 ms (+- 2.5 ms) |
|AGX Xavier|320    |FP16      |C++       |28.1                    |7.4               |10.1 ms (+- 3.2 ms) |
|AGX Xavier|640    |INT8      |Python    |33.2                    |17.3              |27.7 ms (+- 2.0 ms) |
|AGX Xavier|640    |INT8      |C++       |14.5                    |9.5               |15.0 ms (+- 2.7 ms) |

## NOTE
The ***libmyplugins.so*** file in build will be needed for inference. 


# Custom model

For custom model conversion there are some factors to take in consideration. 
-  Only YoloV5 S (small) version is supported.
- You should use your own checkpoint that only contains network weights (i.e. stripped optimizer, which is last output of YoloV5 pipeline after training finishes)
 - Change the ***CLASS_NUM*** in ***yololayer.h*** - ```Line 28``` to number of classes your model has before building yolo. If you've already built it, you can just run ```cmake .. && make``` from build folder after changing the class numbers. 
 - Change the ***Input W*** and ***Input H*** according to resolution you've trained the network in ***yololayer.h*** on - ```Lines 29, 30```
 - Change the ***CONF*** and ***NMS*** thresholds according to your preferences in ***yolov5.cpp*** - ```Lines 12, 13```
 - Change the batch size according to your requirements in ***yolov5.cpp*** - ```Line 14```.

# INT 8 Conversion & Calibration
 Exporting YoloV5 network to INT8 is pretty much straightforward & easy. 

Before you run last command in ***Step 5*** in conversion steps, you should take in consideration that INT8 needs dataset for calibration.
Good news is - we won't need labels, just images. 
 There's no recommended amount of data samples to use for calibration, but as many - as better. 

 In this case, as long as we're exporting the standard yolov5s, trained on COCO dataset, we'll download *val* set of images from coco and do calibration on it. 

1. Enter the ***tensorrt_yolov5*** folder and run. 
```shell
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017 build/coco_calib
```
2. Change the precision in ***yolov5.cpp*** - ```Line 10```

```#define USE_FP16``` -> ```#define USE_INT8```

3. Run the conversion.
```shell
cmake .. && make
sudo ./yolov5 -s yolov5s.wts yolov5s_int8.engine s
```
 It'll do calibration on every image in coco_calib folder, it might take a while. After it's finished, the engine is ready for usage.


## Thanks to the following sources/repositories for scripts & helpful suggestions.
- For TensorRT conversion codebase:
 https://github.com/wang-xinyu/tensorrtx
 - For Torch installation procedures:
 https://qengineering.eu/install-pytorch-on-jetson-nano.html
 - For Swap memory management:
 https://github.com/JetsonHacksNano/installSwapfile
