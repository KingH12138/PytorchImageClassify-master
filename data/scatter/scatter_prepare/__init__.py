import os
from tqdm import tqdm
import pandas as pd


class Classification_scatter_data():
    """

    This class is created to make scatter dataset

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
    def __init__(self, src_dir: str, cls_path: str, refer_path: str):
        self.src_dir = src_dir
        self.cls_path = cls_path
        self.refer_path = refer_path

    def readcls(self,filename):
        """
        这里需要自己改，因为scatter格式数据集的label读入有很多种
        filename -> classname
        :return:class string
        """
        idx = 0
        for string in filename:
            if string<='9' and string>='0':
                break
            else:
                idx += 1
        return filename[:idx]

    def cls_buffer2txt(self):
        cls_list = []
        print("Reading filenames......")
        for filename in tqdm(os.listdir(self.src_dir)):
            label = self.readcls(filename)
            if label in cls_list:
                continue
            else:
                cls_list.append(label)
        print("Done.")
        content = "\n".join(cls_list)
        with open(self.cls_path, 'w') as f:
            f.write(content)

    def cls_txt2buffer(self):
        with open(self.cls_path, 'r') as f:
            cls_list = f.read().split("\n")
            return cls_list

    def generate(self):
        self.cls_buffer2txt()
        print("classes.txt has been saved to " + self.cls_path)
        refer = {"filename": [], "filepath": [], "label": []}
        for filename in os.listdir(self.src_dir):
            filepath = os.path.join(self.src_dir, filename)
            cls_list = self.cls_txt2buffer()
            cls_name = self.readcls(filename)
            label = cls_list.index(cls_name)
            refer['filename'].append(filename)
            refer['filepath'].append(filepath)
            refer['label'].append(label)
        df = pd.DataFrame(data=refer)
        df.to_csv(self.refer_path, encoding='utf-8')
