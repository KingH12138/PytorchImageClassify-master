# Pytorch ImageClassification使用方法


## 一.配置环境

### 

先创建好环境并安装好pytorch，激活环境(至于要安装什么pytorch版本，可以参考[这篇文章](https://blog.csdn.net/Killer_kali/article/details/123173414?spm=1001.2014.3001.5501)：

```

conda create -n torchclassify python=3.8
conda activate torchclassify
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch

```

进入项目目录：

```
cd Pyotrch-ImageClassification-master
```

安装相关包库：

```
pip install -r requirements.txt
```

tips:

如果prettytable库无法安装，可以尝试如下命令：

```
python -m pip install -U prettytable
```

## 二.运行predict测试文件

预测权重下载:https://wwp.lanzouf.com/iQ9g702lytmj

密码:6bbf

将权重存放到任意路径下。
修改predict的-pw参数
在终端中输入:

```
python predict.py -pw your weight's path -pp the path of your image applied to prediciton
```
## 三.制作并训练自己的数据
这里海洋生物4分类数据集为例子，部分数据集下载:


数据下载:https://wwp.lanzouf.com/idxxS02lynkb

密码:67qh

1.在任意目录(推荐是data目录)下放入数据集(JPEGImage)文件夹，图像数据集存储格式不是ImageFloder;

2.运行Multi-class_Weather_Dataset_for_Image_Classification.py脚本，生产datast_info.csv文件;

3.在train.py脚本文集里面修改get_arg函数下的参数；

4.输入:

```
cd tools
```
进入tools文件下，然后输入
```
python train.py -h
```
查看配置参数项，当然你也可以直接在这里修改参数，比如说:
```
python train.py -e 20   # 设置迭代次数epoches为20
```

5.开始训练
```
python train.py
```

## 四.注意事项

2022/5/24日更新:若出现报错“页面太小......”，可以尝试调高datasets下的dataload.py里面的get_dataloader函数的num_workers参数。

2022/5/24日更新:opencv-python仅仅支持全英文路径。
