# -*- coding: utf-8 -*-
"""
Training classification network
Res3D
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
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import time
import dataset.dataload as dataload #加载数据集
import model.Models as Models #加载数据集
import random
import math
import argparse
from torchmetrics.regression import R2Score #求R2
from sklearn.model_selection import KFold
import argparse
import csv

from tensorboardX import SummaryWriter
write = SummaryWriter(log_dir='./result/log')  #将数据存放在这个文件夹中。


parser = argparse.ArgumentParser()
parser.add_argument('--Kcross_num', type=int, default=1) #    #6折交叉，交叉No.
args = parser.parse_args()

Kcross_num = args.Kcross_num  # Shear segment number


def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

def write_data(filenname, data):
    data_head = ['Times','Fx','Fx"','Fz','Fz"','T','T"']
    headers = data_head
    with open(filenname, 'w', newline='') as form:
        writer = csv.writer(form)
        writer.writerow(headers)

    with open(filenname, 'a', newline='') as form:
        writer = csv.writer(form)
        writer.writerows(data)

if __name__ == '__main__':
    setup_seed(0)

    """
    ====================================
    0、Training parameters
    """
    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 40  # 10

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 2

    # Number of training epochs
    num_epochs = 40

    # Learning rate for optimizers
    lr = 0.0001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    length_set = 1

    """
    ====================================
    1、Load data
    """
    Cross_list = [] #6折交叉验证划分，Kcross_num为第几折用于测试
    kf = KFold(n_splits=6, shuffle=True, random_state=0)
    for cross_train, cross_test in kf.split(list(range(0, 12))):
        Cross_list.append([list(cross_train + 1), list(cross_test+1)])

    # Create the test_data
    test_path = r"../Alldata"
    Testset_all = dataload.MyDataset(test_path, time_lenth=length_set,
                                 k_num=Cross_list[Kcross_num - 1][1], noise_mean=0, noise_std_max=0)  # {103620,853}
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
    net = Models.TransformerEncoderModel(input_dim=768, num_heads=4, hidden_dim=256, output_dim=3, num_layers=1, dropout_rate=0.2).to(device)
    net = nn.DataParallel(net, list(range(ngpu)))
    print(net)
    Physical = Models.PhysicalModel()


    """
    ====================================
    6、test
    """
    save_path = './result/checkpoint/1723872378_K1.pth'
    net.load_state_dict(torch.load(save_path))  #save_path
    val_len = 0
    real = torch.empty(0, 3)
    pre = torch.empty(0, 3)
    Accuracy = []
    r2score = R2Score(num_outputs=3, multioutput='raw_values')  # 求R2
    ########################################### validate ###########################################
    net.eval()  # 验证过程中关闭 Dropout
    acc = 0.0
    rmse_sum = 0
    i=1
    with torch.no_grad():
        for test_onedata in test_data:
            print(i)
            i+=1
            sensor,labels = test_onedata# 压力数据

            sensor = sensor.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            #outputs = Physical.frame_caculate(sensor)
            outputs = net(sensor)

            real = torch.cat((real, labels.cpu()), 0)
            pre = torch.cat((pre, outputs.cpu()), 0)
            val_len = val_len + 1
            rmse_sum += torch.sqrt(((outputs - labels) ** 2).sum()).cpu().item()

        test_num = len(test_data) * len(sensor)  # 测试集总数 迭代次数 *batch size

        test_rmse = rmse_sum / test_num
        Accuracy.append(test_rmse)

        test_r2_xyz = r2score(pre, real)  # 求R2
        print(test_r2_xyz)
        test_r2 = test_r2_xyz.sum().cpu().item() / 3

        print('[epoch %d] test_loss: %.3f  test_rmse: %.3f  test_R2: %.3f \n' %
              (0, 0, test_rmse,test_r2))


    """
    ====================================
    7、draw
    """
    print('Start Drawing')
    real_x = real.numpy()
    pre_x = pre.numpy()
    ###画图
    num_X = list(range(len(real_x[:, 0])))
    Times = [x * 0.04 for x in num_X]
    savedata = np.array([Times,real_x[:, 0], pre_x[:, 0], real_x[:, 1], pre_x[:, 1], real_x[:, 2], pre_x[:, 2]])
    savedata = savedata.T
    print(savedata.shape)
    write_data('./outdata_HMIM.csv',savedata )

    for i in range(3):
        plt.figure() #新建图布
        real_x_i = real_x[:, i]
        pre_x_i= pre_x[:, i]

        # 绘图
        plt.plot(Times, real_x_i, label='real_{}'.format(i+1))
        plt.plot(Times, pre_x_i, color='red', linewidth=1, linestyle='--', label='pre_{}'.format(i+1))
        plt.legend()  # 添加图例
        # 展示图形
        savetime = int(time.time())
        plt.savefig('./result/fig/'+str(savetime)+'_'+str(i+1)+'.png')
        plt.show()





