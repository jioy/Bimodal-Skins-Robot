# -*- coding: utf-8 -*-
"""
Training classification network
Fushion Network
==============
**Author**: `zhibin Li`__
"""
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import time
import dataset.dataload as dataload #加载数据集
import model.Models as Models #加载数据集
import random
import math
from sklearn.model_selection import KFold
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import OrderedDict
import csv

from tensorboardX import SummaryWriter
write = SummaryWriter(log_dir='./result/log')  #将数据存放在这个文件夹中。

parser = argparse.ArgumentParser()
parser.add_argument('--Kcross_num', type=int, default=1) #    #6折交叉，交叉No.
parser.add_argument('--length_set', type=int, default=100) #    #length_set.
args = parser.parse_args()

Kcross_num = args.Kcross_num  # Shear segment number
length_set = args.length_set

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True




def get_data():
    """
        ====================================
        0、Training parameters
        """
    # Number of workers for dataloader
    workers = 4

    # Batch size during training
    batch_size = 20  # 10

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 4

    # Number of training epochs
    num_epochs = 15

    # Learning rate for optimizers
    lr = 0.0001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5


    """
    ====================================
    1、Load data
    """
    Cross_list = []  # 6折交叉验证划分，Kcross_num为第几折用于测试
    kf = KFold(n_splits=6, shuffle=True, random_state=0)
    for cross_train, cross_test in kf.split(list(range(0, 12))):
        Cross_list.append([list(cross_train + 1), list(cross_test + 1)])

    # Create the test_data
    test_path = r"../Alldata"
    Testset_all = dataload.MyDataset(test_path, time_lenth=length_set,
                                     k_num=list(range(1, 13)), noise_mean=0, noise_std_max=0)  # {103620,853}
    # 数据集划分
    test_size = int(0.5 * len(Testset_all))
    vali_size = len(Testset_all) - test_size
    validation_dataset, test_dataset = torch.utils.data.random_split(Testset_all, [vali_size, test_size])

    validation_data = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=workers)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)
    """
    ====================================
    2、Load model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the generator
    net = Models.Res3D().to(device)
    print(net)

    state_dict = torch.load('./result/checkpoint/1718260974_K1_L100.pth')
    # more GPU
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 去掉 `module.`
        # name = k.replace(“module.", "")
        state_dict_new[name] = v

    net.load_state_dict(state_dict_new)


    """
    ====================================
       3、Feature extraction
    """

    return_layers = {'avg_pool': 'feature_512'}
    backbone = torchvision.models._utils.IntermediateLayerGetter(net, return_layers)

    backbone.eval()
    test_sensor = torch.rand(10, 1, 300, 32, 24)
    test_sensor = test_sensor.to(device, dtype=torch.float)  # 转换统一数据格式
    out = backbone(test_sensor)

    """
    ====================================
    3、Test
    """
    net.eval()

    data_state = 0
    cnt = 0
    for test_onedata in test_data:
        cnt = cnt + 1
        print(cnt)
        test_sensor, test_labels = test_onedata
        test_sensor = test_sensor.view(test_sensor.size(0),1,test_sensor.size(1),32,24)
        test_labels = test_labels.view(test_labels.size(0))
        test_sensor = test_sensor.to(device, dtype=torch.float)  # 转换统一数据格式
        test_labels = test_labels.to(device, dtype=torch.long)

        out = backbone(test_sensor)
        feature_512 = out['feature_512']

        if (data_state == 0):
            outdata = feature_512.cpu()
            out_label = test_labels.cpu()
            data_state = 1
        else:
            outdata = torch.cat((outdata, feature_512.cpu()), 0)
            out_label = torch.cat((out_label, test_labels.cpu()), 0)



    outdata = outdata.view(outdata.size(0), -1)
    outdata = outdata.detach().numpy()
    out_label = out_label.detach().numpy()
    print(outdata.shape)
    print(out_label.shape)

    return outdata, out_label


def write_data(filenname, data):
    data_head = ['X','Y']
    headers = data_head
    with open(filenname, 'w', newline='') as form:
        writer = csv.writer(form)
        writer.writerow(headers)

    with open(filenname, 'a', newline='') as form:
        writer = csv.writer(form)
        writer.writerows(data)


if __name__ == '__main__':
    setup_seed(0)
    datas , labels = get_data()

    tsne = TSNE(n_components=2, perplexity=120, init='pca', random_state=0)
    result = tsne.fit_transform(datas, labels)
    ax = plt.subplot()
    ax.set_title('2D t-SNE of tactile features')
    scatter = ax.scatter(result[:, 0], result[:, 1], c=labels)
    legendClass = ax.legend(*scatter.legend_elements(prop="colors"),
                            loc="upper left", title="classes")
    ax.add_artist(legendClass)
    plt.colorbar(scatter)

    savedata = np.array([labels, result[:, 0], result[:, 1]])
    savedata = savedata.T
    write_data('./result/fig/Tsne.csv',savedata )

    plt.savefig('./result/fig/tsne.png', dpi=600, bbox_inches='tight')
    plt.show()




