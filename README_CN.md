# Yolov5在Nvidia Jetson设备上的模型转换与部署 [Ultralytics EXPORT]
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://gitee.com/jario-jin/yolov5-on-nvidia-jetson/blob/master/LICENSE)
[![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)]()

[<img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash-export-competition.png">](https://github.com/ultralytics/yolov5/discussions/3213)


# 说明
该代码库用于将 Yolov5 部署到 NVIDIA Jetson 平台并通过 TensorRT 技术对其进行加速。
可用设备，如 Nano (472 GFLOPS)、TX2 (1.33 TFLOPS)、Xavier NX (21 TOPS) 和 AGX Xavier (32TOPS)。

# JetPack
首先，您应该从 NVIDIA 获得最新的 JetPack *v4.5.1* 并将其部署到 Jetson Nano。您可以找到 JetPack 下载选项 **[此处](https://developer.nvidia.com/embedded/jetpack)** 


# FP16模型转换与部署
1. 下载该代码库并进入目录:
   ```shell
   git clone https://gitee.com/jario-jin/yolov5-on-nvidia-jetson.git
   cd yolov5-on-nvidia-jetson
   ```
2. 安装所需环境库:
   ```shell
   bash scripts/install_requirements_cn.sh
   ```
3. 转换模型:
   ```shell
   bash scripts/convert_models_cn.sh
   ```
4. 在COCO评估集上测试精度与速度:
   ```shell
   wget http://jario.ren/dataset/val2017.zip
   wget http://jario.ren/dataset/instances_val2017.json
   unzip val2017.zip
   # 方式一： 使用Python前端推断
   python3 inference.py --input "val2017/*.jpg" --gt instances_val2017.json
   # 方式二： 使用C++前端推断
   bash scripts/cpp_tensorrt_coco_eval.sh
   ```
# 精度与速度测试结果

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

## 注意
在推断过程中需要 ***libmyplugins.so*** 文件。


# 自定义模型

对于自定义模型转换，需要考虑如下一些因素。
-  目前只有 Yolov5s (small) 是可用的。
- 应该使用仅包含网络权重的checkpoint（即剥离优化器，这是训练完成后 Yolov5 的最后输出）。
 - 将 ***yololayer.h*** - ```Line 28``` 中的 ***CLASS_NUM*** 更改为您的模型在构建 Yolo 之前的类别数。
 - 根据您在 ***yololayer.h*** - ```Lines 29, 30``` 中训练网络的分辨率更改 ***Input W*** 和 ***Input H***。
 - 根据您在 ***yolov5.cpp***- ```Lines 12, 13``` 中的偏好更改 ***CONF*** 和 ***NMS*** 阈值。
 - 在 ***yolov5.cpp*** - ```Line 14``` 中根据您的要求更改批大小。

# INT8模型转换与标定
将 Yolov5 网络导出到 INT8 非常简单易行，但您应该考虑到 INT8 需要数据集进行校准。
好消息是 - 我们不需要标签，只需要图像。
没有推荐的用于校准的数据样本量，但尽可能多 - 更好。

在这种情况下，导出在 COCO 数据集上训练的标准 yolov5s，则可以从 COCO 下载 *val* 图像集并对其进行校准。

1. 进入 ***yolov5-on-nvidia-jetson*** 文件夹并运行: 
```shell
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017 build/coco_calib
```
2. 修改 ***yolov5.cpp*** - ```Line 10```中的模型精度

```#define USE_FP16``` -> ```#define USE_INT8```

3. 运行转换程序:
```shell
cmake .. && make
sudo ./yolov5 -s yolov5s.wts yolov5s_int8.engine s
```
注意：对 `coco_calib` 文件夹中的每个图像进行校准，需要一段时间。


## 感谢以下开源仓库的代码和有用的建议。
- TensorRT模型转换:
 https://github.com/wang-xinyu/tensorrtx
 - Torch在Jetson设备上的安装:
 https://qengineering.eu/install-pytorch-on-jetson-nano.html
 - 交换区内存管理:
 https://github.com/JetsonHacksNano/installSwapfile
