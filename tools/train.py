import argparse
from datetime import datetime
import sys

sys.path.append('..')
sys.path.append('.')

import torch
from prettytable import PrettyTable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets.dataload import get_dataloader
from log_generator import log_generator
from model.resnet import Resnet18

# ----------------------------------------------------------------------------------------------------------------------
# 导入参数
def get_arg():
    parser = argparse.ArgumentParser(description='classification parameter configuration(train)')
    parser.add_argument(
        '-t',
        type=str,
        default='pytorch-imageclassification-master',
        help='This is your task theme name'
    )
    parser.add_argument(
        '-imagep',
        type=str,
        default=r'..\data\Multi-class_Weather_Dataset_for_Image_Classification\JPEGImage',
        help="image's directory"
    )
    parser.add_argument(
        '-csvp',
        type=str,
        default=r'..\data\Multi-class_Weather_Dataset_for_Image_Classification\dataset_info.csv',
        help="DIF(dataset information file)'s path"
    )
    parser.add_argument(
        '-tp',
        type=float,
        default=0.9,
        help="train dataset's percent"
    )
    parser.add_argument(
        '-bs',
        type=int,
        default=16,
        help="train dataset's batch size"
    )
    parser.add_argument(
        '-rs',
        type=tuple,
        default=(224, 224),
        help='resized shape of input tensor'
    )
    parser.add_argument(
        '-classes',
        type=list,
        default=['shine', 'rain', 'cloudy', 'sunrise'],
        help="classes.txt's path"
    )
    parser.add_argument(
        '-cn',
        type=int,
        default=4,
        help='the number of classes'
    )
    parser.add_argument(
        '-e',
        type=int,
        default=2,
        help='epoch'
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=0.001,
        help='learning rate'
    )
    parser.add_argument(
        '-ld',
        type=str,
        default='../workdir',
        help="the training log's save directory"
    )

    return parser.parse_args()


args = get_arg()  # 得到参数Namespace
# ----------------------------------------------------------------------------------------------------------------------
print("Training device information:")
# 训练设备信息
device_table = ""
if torch.cuda.is_available():
    device_table = PrettyTable(['number of gpu', 'applied gpu index', 'applied gpu name'], min_table_width=80)
    gpu_num = torch.cuda.device_count()
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name()
    device_table.add_row([str(gpu_num), str(gpu_index), str(gpu_name)])
    print('{}\n'.format(device_table))
else:
    print("Using cpu......")
    device_table = 'CPU'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------------------------------------------
# 数据集信息
print("Use dataset information file:{}\nLoading dataset from path: {}......".format(args.csvp, args.imagep))
train_dl, valid_dl, samples_num, train_num, valid_num = get_dataloader(args.imagep, args.csvp, args.rs, args.bs
                                                                       , args.tp)
dataset_table = PrettyTable(['number of samples', 'train number', 'valid number', 'percent'], min_table_width=80)
dataset_table.add_row([samples_num, train_num, valid_num, args.tp])
print("{}\n".format(dataset_table))
# ----------------------------------------------------------------------------------------------------------------------
# 训练组件配置
print("Classes information:")
classes_table = PrettyTable(args.classes, min_table_width=80)
classes_table.add_row(range(len(args.classes)))
print("{}\n".format(classes_table))
print("Train information:")
model = Resnet18(pretrain=True, num_classes=args.cn).to(device)
optimizer = Adam(params=model.parameters(), lr=args.lr)
loss_fn = CrossEntropyLoss()
train_table = PrettyTable(['theme', 'resize', 'batch size', 'epoch', 'learning rate', 'directory of log'],
                          min_table_width=120)
train_table.add_row([args.t, args.bs, args.bs, args.e, args.lr, args.ld])
print('{}\n'.format(train_table))
# ----------------------------------------------------------------------------------------------------------------------
# 开始训练
losses = []
accuracies = []
precisions = []
recalls = []
f1s = []

st = datetime.now()
for epoch in range(args.e):
    model.train()
    train_bar = tqdm(iter(train_dl), ncols=150, colour='red')
    train_loss = 0.
    i = 0
    for train_data in train_bar:
        x_train, y_train = train_data
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        output = model(x_train)
        loss = loss_fn(output, y_train)
        optimizer.zero_grad()
        # clone().detach()：可以仅仅复制一个tensor的数值而不影响tensor# 原内存和计算图
        train_loss += loss.clone().detach().cpu().data
        loss.backward()
        optimizer.step()
        # 显示每一批次的loss
        train_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(train_dl)))
        train_bar.set_postfix({"train loss": "%.3f" % loss.data})
        i += 1
    train_loss = train_loss / i
    # 最后得到的i是一次迭代中的样本数批数
    losses.append(train_loss)

    model.eval()
    valid_bar = tqdm(iter(valid_dl), ncols=150, colour='red')
    valid_acc = 0.
    valid_pre = 0.
    valid_recall = 0.
    valid_f1 = 0.
    i = 0
    for valid_data in valid_bar:
        x_valid, y_valid = valid_data
        x_valid = x_valid.to(device)
        y_valid_ = y_valid.clone().detach().numpy()  # y_valid就不必放到gpu上训练了
        output = model(x_valid)  # shape:(N*cls_n)
        output_ = output.clone().detach().cpu()
        _, pred = torch.max(output_, 1)  # 输出每一行(样本)的最大概率的下标
        pred_ = pred.clone().detach().numpy()
        acc, precision, recall, f1 = accuracy_score(y_true=y_valid_, y_pred=pred_), \
                                     precision_score(y_true=y_valid_, y_pred=pred_, average='weighted'), \
                                     recall_score(y_true=y_valid_, y_pred=pred_, average='weighted'), \
                                     f1_score(y_true=y_valid_, y_pred=pred_, average='weighted')
        valid_acc = valid_acc + acc
        valid_pre = valid_pre + precision
        valid_recall = valid_recall + recall
        valid_f1 = valid_f1 + f1
        # 显示每一批次的acc/precision/recall/f1
        valid_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(valid_dl)))
        valid_bar.set_postfix({"accuracy": "%.3f" % acc, "precision": "%.3f" % precision, "recall": "%.3f" % recall,
                               "f1": "%.3f" % f1})
        i += 1

    # 最后得到的i是一次迭代中的样本数批数
    valid_acc = valid_acc / i
    valid_pre = valid_pre / i
    valid_recall = valid_recall / i
    valid_f1 = valid_f1 / i
    accuracies.append(valid_acc)
    precisions.append(valid_pre)
    recalls.append(valid_recall)
    f1s.append(valid_f1)
et = datetime.now()

log_generator(args.t, et - st, dataset_table, classes_table, device_table, train_table, optimizer, model, args.e,
              [losses, accuracies, precisions, recalls, f1s], args.ld)
