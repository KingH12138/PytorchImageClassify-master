import os
from datetime import datetime
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from math import sqrt


def generate_feature(img_path, resize, model_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    layers = list(list(model.children())[0].named_children())
    transformer = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])
    with torch.no_grad():
        pil_img = Image.open(img_path)
        img_input = transformer(pil_img)
        img_input = img_input.reshape((1, *img_input.shape))
        for i in tqdm(range(len(layers))):
            try:
                layer = layers[:i + 1]
                layer_ = torch.nn.Sequential(*[i[1] for i in layer])
                name = layers[i][0]
                outputs = layer_(img_input.to(device))  # 1,c,h,w
                outputs = outputs.permute(1, 0, 2, 3)  # c,1,h,w
                shape = outputs.shape
                grid_tensor = make_grid(outputs, nrow=int(sqrt(shape[0])))
                save_image(grid_tensor, '{}/{}.jpg'.format(save_dir, name))
            except:
                print("{} can not be plot normally.".format(name))


def get_confusion_matrix(y_pred, y_label, cls_path, fig_save_dir=None):
    #############
    # TP    FN  #
    # FP    TN  #
    #############
    """
    :param y_pred: a prediction numpy array just like [0,1,3,5,7]
    :param y_label:a label numpy array just like [0,1,3,5,7]
    :return:None
    """
    sns.set_style("whitegrid")
    with open(cls_path, 'r') as f:
        cls_list = f.read().split('\n')[:-1]
    cmatrix = confusion_matrix(y_label, y_pred, labels=cls_list)
    df = pd.DataFrame(data=cmatrix, index=cls_list, columns=cls_list)
    plt.figure(dpi=300, figsize=(4, 3))
    plt.rc('font', family='Times New Roman', size=14)
    sns.heatmap(data=df)
    if fig_save_dir:
        plt.savefig(fig_save_dir + '/heatmap.svg')
    return cmatrix


def dataset_distribution(refer_path, cls_path, save_dir=None):
    """
    tips:make sure your refer.csv have "label" key.
    """
    sns.set_style("whitegrid")
    df = pd.read_csv(refer_path, encoding='utf-8')
    cls_list = open(cls_path, 'r').read().split('\n')[:-1]
    plt.figure(dpi=300, figsize=(4, 3))
    plt.rc('font', family='Times New Roman', size=14)
    count_df = pd.DataFrame(df['label'].value_counts()).T
    cols = list(count_df.columns)
    cols = [cls_list[i] for i in cols]
    count_df.columns = cols
    sns.barplot(data=count_df)
    if save_dir:
        plt.savefig(save_dir + '/dataset_distribution.svg')


def torch2onnx(model_path, outputs_path, inputs_shape, device='cuda'):
    """
    tips:你保存的模型一定是直接通过torch.save(model)保存的——也就是保存整个模型
    :param model_path:.pth文件路径
    :param outputs_path:输出的.onnx文件路径
    :param inputs_shape:(bs,x,x,h,w)
    :param device:cuda or cpu
    :return:
    """
    dummy_input = torch.randn(*inputs_shape, device=device)
    input_names = ["input"]
    output_names = ["output"]
    model = torch.load(model_path)
    torch.onnx.export(model, dummy_input, outputs_path, verbose=True, input_names=input_names,
                      output_names=output_names)


def df_generator(epoches, tags, save_path=None):
    """
    2022/5/24:update Auc and mAP
    """
    keys = ['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1']
    values = [range(1, epoches + 1)] + tags
    data = dict(zip(keys, values))
    df = pd.DataFrame(data=data)
    if save_path:
        df.to_csv(save_path, encoding='utf-8')
    return df


def log_plot(epoches, tags, save_fig_dir, csv_save_path=None):
    """
    :param epoches:迭代次数
    :param tags:[loss,acc,precision,recall,f1]
    :param save_fig_path:保存路径
    """
    df = df_generator(epoches, tags, save_path=csv_save_path)
    sns.set_style("whitegrid")
    indicators = list(df.columns[2:])
    data = df.to_numpy()[:, 2:]
    index = list(df.index)
    df_ = pd.DataFrame(data=data, index=index, columns=indicators)
    plt.rc('font', family='Times New Roman', size=14)
    plt.figure(dpi=300, figsize=(4, 3))
    plt.xlabel("Epoch")
    plt.ylabel("Indicator")
    sns.lineplot(data=df_, markers=True)
    plt.savefig(save_fig_dir + '/indicators.svg')

    plt.figure(dpi=300, figsize=(4, 3))
    sns.lineplot(data=df, y='Loss', x="Epoch")
    plt.savefig(save_fig_dir + '/loss_epoch.svg')
    # ------------------------------------
    """
    子图代码
    """
    # rows = int(sqrt(len(indicators)))
    # cols = int(len(indicators)/rows)+1
    # plt.subplots(nrows=rows,ncols=cols)
    # for i in range(len(indicators)):
    #     plt.subplot(rows,cols,i+1)
    #     sns.lineplot(data=df,x='Epoch',y=indicators[i])
    # plt.subplots_adjust(wspace=0.3, hspace=0.6)
    # ------------------------------------


def log_generator(train_theme_name, csv_path, cls_path, inputs_shape, duration,
                  dataset_info_table, classes_info_table,
                  training_device_table, training_info_table,
                  optimizer, model, epoches,
                  tags, log_save_dir, best_cp, test_img_path=None, resize=None):
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
folder information:\n{}\n
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
    checkpoints_path = exp_path + '/checkpoints'
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    # logger
    log_name = '{}_{}_{}_{}_{}_{}.log'.format(
        train_theme_name,
        year, month, day,
        hour, minute, second)
    log_file = open(exp_path + '/' + log_name, 'w', encoding='utf-8')
    log_file.write(content)
    log_file.close()
    # checkpoints
    # final_checkpoints
    torch.save(model, checkpoints_path + '/' + 'final_model.pth'.format(
        train_theme_name,
        year, month, day, hour,
        minute, second
    ))
    torch2onnx(
        model_path=checkpoints_path + '/' + 'final_model.pth'.format(
            train_theme_name,
            year, month, day, hour,
            minute, second
        ),
        outputs_path=checkpoints_path + '/' + 'final_model.onnx',
        inputs_shape=inputs_shape
    )
    # best_checkpoints
    best_cp_path = checkpoints_path + '/' + 'best_f1.pth'.format(
        train_theme_name,
        year, month, day, hour,
        minute, second
    )
    torch.save(best_cp, best_cp_path)
    torch2onnx(
        model_path=checkpoints_path + '/' + 'best_f1.pth'.format(
            train_theme_name,
            year, month, day, hour,
            minute, second
        ),
        outputs_path=checkpoints_path + '/' + 'best_f1.onnx',
        inputs_shape=inputs_shape
    )
    # indicator.csv and indicator.jpg
    log_plot(epoches, tags, save_fig_dir=exp_path, csv_save_path=exp_path + '/indicators.csv')
    print("Training log has been saved to path:{}".format(exp_path))
    # datasets' distribution
    dataset_distribution(refer_path=csv_path, cls_path=cls_path, save_dir=exp_path)
    # feature map
    if test_img_path != None and resize != None:
        generate_feature(test_img_path, resize, best_cp_path, exp_path + "/feature_map")
