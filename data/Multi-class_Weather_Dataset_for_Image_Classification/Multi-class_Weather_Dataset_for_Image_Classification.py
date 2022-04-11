"""

This python demo is created to make "Multi-class_Weather_Dataset_for_Image_Classification" dataset

easy to be processed by my demo.

image_dir format:

-JPEGImage
    -classname+index.jpg
    -.....
csv format:

index   filename filepath label
0       ...     ...     ...
1       ...     ...     ...
2       ...     ...     ...
3       ...     ...     ...
......

"""
import os

import cv2
import numpy as np
import pandas as pd


# 好像有杂糅图片在这里面，其通道数为1，改一下
def check_in():
    image_path = './JPEGImage'
    error = []
    for filename in os.listdir(image_path):
        path = image_path + '/{}'.format(filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        try:
            if img.shape[-1] != 3:
                error.append(path)
        except:
            # print(path)
            error.append(path)
    print(error)
    for path in error:
        os.remove(path)


# 写一个函数方便把classname和index分开
def split_classname(filename):
    num_start = 0
    for i in range(len(filename)):
        if ord(filename[i]) >= 48 and ord(filename[i]) <= 57:
            num_start = i
            break
    return filename[:num_start]


# 生成classes.txt文件
def generate_classes_txt():
    image_path = './JPEGImage'
    classes = []
    for name in os.listdir(image_path):
        filename = name[:-4]
        classes.append(split_classname(filename))
    classes = list(set(classes))
    with open('./classes.txt', 'w') as f:
        content = ""
        for i, classes_name in enumerate(classes):
            content = content + "{} {}\n".format(i, classes_name)
        f.write(content)


def read_classes():
    path = './classes.txt'
    content = ""
    with open(path, 'r') as f:
        content = f.read()
    content = content.split("\n")[:-1]  # 最后一个是空列表，不需要
    output = []
    for item in content:
        item_list = item.split(" ")
        output.append(item_list[1])  # list
    return output


def multi_weather_csv():
    image_path = r'{}/JPEGImage'.format(os.getcwd())
    if os.path.exists('./classes.txt') == False:
        generate_classes_txt()
    classes = read_classes()
    infomation_array = []  # shape=(n,3)
    for name in os.listdir(image_path):
        filename = name[:-4]
        path = image_path + '/{}'.format(name)
        class_name = split_classname(filename)
        infomation_array.append([filename, path, classes.index(class_name)])
    info_arr = np.array(infomation_array)
    col = ['filename', 'filepath', 'label']
    df = pd.DataFrame(info_arr, columns=col)
    df.to_csv('./dataset_info.csv', encoding='utf-8')


# check_in()
multi_weather_csv()
