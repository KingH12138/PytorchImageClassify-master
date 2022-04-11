import argparse

import cv2
import torch
from PIL.Image import open
from torchvision import transforms

from model.resnet import Resnet18


def get_arg():
    parser = argparse.ArgumentParser(description='classification parameter configuration(predict)')
    parser.add_argument(
        '-pw',
        type=str,
        default=r'..\test_weight.pth',
        help='the weight applied to predict'
    )
    parser.add_argument(
        '-pp',
        type=str,
        default=r'..\demo.jpg',
        help="prediction image's path"
    )
    parser.add_argument(
        '-classes',
        type=list,
        default=['shine', 'rain', 'cloudy', 'sunrise'],
        help="classes.txt's path"
    )
    parser.add_argument(
        '-rs',
        type=tuple,
        default=(224,224),
        help='resized shape of input tensor'
    )
    return parser.parse_args()

args = get_arg()
# ----------------------------------------------------------------------------------------------------------------------
# 对预测图片进行预处理-resize+totensor
transformer = transforms.Compose([
    transforms.Resize(args.rs),
    transforms.ToTensor(),
])
pil_img = open(args.pp)
img_input = transformer(pil_img)
img_input = img_input.reshape((1, *img_input.shape))
# ----------------------------------------------------------------------------------------------------------------------
# 加载好模型
if torch.cuda.is_available():
    print("Predict on cuda and there are/is {} gpus/gpu all.".format(torch.cuda.device_count()))
    print("Device name:{}\nCurrent device index:{}.".format(torch.cuda.get_device_name(), torch.cuda.current_device()))
else:
    print("Predict on cpu.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Resnet18(num_classes=4)
print("Load weight from the path:{}.".format(args.pw))
model.load_state_dict(torch.load(args.pw))
model = model.to(device)
img_input = img_input.to(device)
# ----------------------------------------------------------------------------------------------------------------------
# 前向传播进行预测
output = model(img_input)
score, prediction = torch.max(output, dim=1)
pred_class = args.classes[prediction.reshape((1,))]
score_value = score.detach().cpu().numpy().tolist()[0]
# ----------------------------------------------------------------------------------------------------------------------
# 进行展示
show_arr = cv2.imread(args.pp, cv2.IMREAD_UNCHANGED)
cv2.putText(show_arr, '%s:%.3f' % (pred_class, score_value), (0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            color=(255, 0, 0))  # 默认使用蓝色
cv2.imshow('classify-prediction', show_arr)
cv2.waitKey(0)