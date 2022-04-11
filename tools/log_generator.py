import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


def log_plot(epoches, tags, save_fig_path=None):
    """
    :param epoches:迭代次数
    :param tags:[loss,acc,precision,recall,f1]
    :param save_fig_path:保存路径
    """
    plt.figure(figsize=(11, 7))
    # loss
    plt.subplot(2, 3, 1)
    plt.plot(np.arange(1, epoches + 1), tags[0], c=(1, 0, 0))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss-Epoch')
    # accuracy
    plt.subplot(2, 3, 2)
    plt.plot(np.arange(1, epoches + 1), tags[1], c=(0, 1, 0))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-Epoch')
    # precisions
    plt.subplot(2, 3, 3)
    plt.plot(np.arange(1, epoches + 1), tags[2], c=(0, 0, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision-Epoch')
    # recall
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(1, epoches + 1), tags[3], c=(0.75, 0.25, 0))
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall-Epoch')
    # f1
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(1, epoches + 1), tags[4], c=(0, 0.75, 0.25))
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('F1-Epoch')

    plt.savefig(save_fig_path)


def log_generator(train_theme_name, duration,
                  dataset_info_table, classes_info_table,
                  training_device_table, training_info_table,
                  optimizer, model, epoches,
                  tags, log_save_dir):
    nowtime = datetime.now()
    year = str(nowtime.year)
    month = str(nowtime.month)
    day = str(nowtime.day)
    hour = str(nowtime.hour)
    minute = str(nowtime.minute)
    second = str(nowtime.second)
    nowtime_strings = year + '/' + month + '/' + day + '/' + hour + ':' + minute + ':' + second
    workplace_path = os.getcwd()
    content = """
Theme:{}\n
Date:{}\n
Time used:{}\n
workplace:{}\n
dataset information:\n{}\n
classes:\n{}\n
training device:\n{}\n
training basic configuration:\n{}\n
Optimizer:\n{}\n
Model:\n{}\n,
    """.format(
        train_theme_name,
        nowtime_strings,
        duration,
        workplace_path,
        dataset_info_table,
        classes_info_table,
        training_device_table,
        training_info_table,
        str(optimizer),
        str(model)
    )
    exp_name = 'exp-{}_{}_{}_{}_{}_{}'.format(
        train_theme_name,
        year, month, day,
        hour, minute, second)
    exp_path = log_save_dir + '/' + exp_name
    if os.path.exists(exp_path) == 0:
        os.makedirs(exp_path)
    log_name = '{}_{}_{}_{}_{}_{}.log'.format(
        train_theme_name,
        year, month, day,
        hour, minute, second)
    file = open(exp_path + '/' + log_name, 'w', encoding='utf-8')
    file.write(content)
    file.close()
    torch.save(model.state_dict(), exp_path + '/' + '{}_{}_{}_{}_{}_{}.pth'.format(
        train_theme_name,
        year, month, day, hour,
        minute, second
    ))
    log_plot(epoches, tags, save_fig_path=exp_path + '/indicators.jpg')
    print("Training log has been saved to path:{}".format(exp_path))
