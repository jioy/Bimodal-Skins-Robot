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
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import time
import dataset.dataload as dataload
import model.Models as Models
import random
import math
import argparse
from torchmetrics.regression import R2Score #R2
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
import argparse
from tqdm import tqdm


from tensorboardX import SummaryWriter
write = SummaryWriter(log_dir='./result/log')  #outdata log


parser = argparse.ArgumentParser()
parser.add_argument('--Kcross_num', type=int, default=1) # Kcross No.
args = parser.parse_args()

Kcross_num = args.Kcross_num  # Shear segment number


def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True


def calculate_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return r2.item()



if __name__ == '__main__':
    setup_seed(0)

    """
    ====================================
    0、Training parameters
    """
    # Number of workers for dataloader
    workers = 4

    # Batch size during training
    batch_size = 40  # 10

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 4

    # Number of training epochs
    num_epochs = 5

    # Learning rate for optimizers
    lr = 0.00001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    #切割时间长度
    length_set = 1

    """
    ====================================
    1、Load data
    """
    Cross_list = [] #Kcross_num list
    kf = KFold(n_splits=6, shuffle=True, random_state=0)
    for cross_train, cross_test in kf.split(list(range(0, 12))):
        Cross_list.append([list(cross_train + 1), list(cross_test+1)])
    train_path = r"../Alldata"
    trainset = dataload.MyDataset(train_path,time_lenth = length_set,
                                  k_num = Cross_list[Kcross_num-1][0], noise_mean = 0, noise_std_max = 1) # {103620,853}


    # Create the train_data
    train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)



    # Create the test_data
    test_path = r"../Alldata"
    Testset_all = dataload.MyDataset(test_path,time_lenth = length_set,
                                 k_num = Cross_list[Kcross_num-1][1], noise_mean = 0, noise_std_max = 0)  # {103620,853}
    # TESTdataset vali and test
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
    #net = Models.MLP_Net().to(device)
    net = Models.TransformerEncoderModel(input_dim=768, num_heads=4, hidden_dim=256, output_dim=3, num_layers=1, dropout_rate=0.2).to(device)
    net = nn.DataParallel(net, list(range(ngpu)))
    Physical = Models.PhysicalModel()
    print(net)


    """
    ====================================
    3、Initial set
    """

    # Loss Functions and Optimizers
    loss_function = nn.MSELoss()

    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    cuda =  True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    savetime = int(time.time())
    save_path = './result/checkpoint/'+ str(savetime) + '_K' + str(Kcross_num) + '.pth'
    best_rmse = 10000
    best_r2 = 0
    Accuracy = []


    """
    ====================================
    4、Train
    """
    lamda_a = 0.1
    iteration = 0
    r2score = R2Score(num_outputs=3, multioutput='raw_values')  # 求R2
    print(device)
    for epoch in range(num_epochs):
        ########################################## train ###############################################

        net.train()
        running_loss = 0.0
        time_start = time.perf_counter()
        # 训练集进度条
        train_loader_tqdm = tqdm(train_data, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for step, data in enumerate(train_loader_tqdm):
            iteration = iteration + 1
            sensor,labels = data #pressure data
            sensor = sensor.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)

            optimizer.zero_grad()  # 清除历史梯度
            Y2 = Physical.frame_caculate(sensor).float()
            outputs = net(sensor)  # 正向传播

            #train_loss = (1-lamda_a) * loss_function(outputs,labels) + lamda_a*loss_function(outputs,Y2)
            train_loss = loss_function(outputs, labels)
            train_loss.backward(retain_graph=True)  # 反向传播
            optimizer.step()  # 优化器更新参数

            train_r2 = calculate_r2(labels, outputs)

            write.add_scalar('train_loss', train_loss.item(),
                             iteration)  # TODO:  每进行完一个iteration，可视化loss.
            write.add_scalar('train_R2', train_r2,
                             iteration)  # TODO:  每进行完一个iteration，可视化R2.

            # 更新进度条描述
            train_loader_tqdm.set_postfix({
                'Train Loss': f'{train_loss.item():.4f}',
            })


        """
            ====================================
            5、eval
            """
        ########################################### validate ###########################################
        net.eval()  # 验证过程中关闭 Dropout

        acc = 0.0
        rmse_sum = 0
        real = torch.empty(0, 3)
        pre = torch.empty(0, 3)
        with torch.no_grad():
            for val_data in validation_data:

                sensor,labels = val_data  # snesor data

                sensor = sensor.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                Y2 = Physical.frame_caculate(sensor)
                outputs = net(sensor)

                val_loss = (1 - lamda_a) * loss_function(outputs, labels) + lamda_a * loss_function(outputs, Y2)

                real = torch.cat((real, labels.cpu()), 0)
                pre = torch.cat((pre, outputs.cpu()), 0)


                rmse_sum += torch.sqrt(((outputs - labels) ** 2).sum()).cpu().item()

            val_num = len(validation_data) * len(sensor)  #测试集总数 :iter num(81) * batch size(256)

            val_rmse = rmse_sum / val_num
            Accuracy.append(val_rmse)

            val_r2_xyz = r2score(pre, real) #求R2
            print(val_r2_xyz)
            val_r2 = val_r2_xyz.sum().cpu().item()/3

            #保存准确率最高的那次网络参数
            if val_r2 > best_r2:
                best_r2 = val_r2
                torch.save(net.state_dict(), save_path)

            print('[epoch %d] train_loss: %.3f  validation_rmse: %.3f \n' %
                  (epoch + 1, running_loss / step, val_rmse))


    print('Start Testing')

    """
    ====================================
    6、test
    """
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

            Y2 = Physical.frame_caculate(sensor)
            outputs = net(sensor)
            real = torch.cat((real, labels.cpu()), 0)
            pre = torch.cat((pre, outputs.cpu()), 0)
            val_len = val_len + 1
            rmse_sum += torch.sqrt(((outputs - labels) ** 2).sum()).cpu().item()

        test_num = test_dataset.__len__()  # 测试集总数 迭代次数 *batch size

        test_rmse = rmse_sum / test_num
        Accuracy.append(test_rmse)

        test_r2_xyz = r2score(pre, real)  # 求R2
        print(test_r2_xyz)
        test_r2 = test_r2_xyz.sum().cpu().item() / 3

        print('[epoch %d] test_loss: %.3f  test_rmse: %.3f  test_R2: %.3f \n' %
              (0, 0, test_rmse,test_r2))

    write.close()





