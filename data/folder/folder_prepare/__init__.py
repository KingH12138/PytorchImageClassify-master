import os
from tqdm import tqdm
import pandas as pd


class Classification_folder_data():
    """

    This class is created to make folder dataset

    easy to be processed by my demo.

    image_dir format:

    -JPEGImage
        -classname0
            -index0.jpg
            -index1.jpg
            ......
        -classname1
        ......
    csv format:

    index   filename filepath label
    0       ...     ...     ...
    1       ...     ...     ...
    2       ...     ...     ...
    3       ...     ...     ...
    ......

    """
    def __init__(self, src_dir:str,cls_path:str,refer_path:str):
        self.src_dir = src_dir
        self.cls_path = cls_path
        self.refer_path = refer_path

    def cls_buffer2txt(self):
        cls_list = os.listdir(self.src_dir)
        content = "\n".join(cls_list)
        with open(self.cls_path,'w') as f:
            f.write(content)

    def cls_txt2buffer(self):
        with open(self.cls_path,'r') as f:
            cls_list = f.read().split("\n")
            return cls_list

    def generate(self):
        self.cls_buffer2txt()
        print("classes.txt has been saved to " + self.cls_path)
        refer = {"filename":[],"filepath":[],"label":[]}
        for cls_name in tqdm(os.listdir(self.src_dir)):
            cls_path = os.path.join(self.src_dir, cls_name)
            for filename in os.listdir(cls_path):
                filepath = os.path.join(cls_path,filename)
                cls_list = self.cls_txt2buffer()
                label = cls_list.index(cls_name)
                refer['filename'].append(filename)
                refer['filepath'].append(filepath)
                refer['label'].append(label)
        df = pd.DataFrame(data=refer)
        df.to_csv(self.refer_path,encoding='utf-8')










