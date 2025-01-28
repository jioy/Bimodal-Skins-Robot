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
import dataset.dataload as dataload #加载数据集
import model.Models as Models #load Model
import model.MobileNet as MobileNet #load Model
import random
import math
import argparse
from torchmetrics.regression import R2Score #求R2
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
import argparse



from tensorboardX import SummaryWriter
write = SummaryWriter(log_dir='./result/log')  #将数据存放在这个文件夹中。

parser = argparse.ArgumentParser()
parser.add_argument('--Kcross_num', type=int, default=1) #    #6折交叉，交叉No.
parser.add_argument('--length_set', type=int, default=300) #    #6折交叉，交叉No.
args = parser.parse_args()

Kcross_num = args.Kcross_num  # Shear segment number
length_set = args.length_set



def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True



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
    num_epochs = 10

    # Learning rate for optimizers
    lr = 0.001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    """
    ====================================
    1、Load data
    """
    Cross_list = [] #6折交叉验证划分，Kcross_num为第几折用于测试
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
    testset = dataload.MyDataset(test_path,time_lenth = length_set,
                                 k_num = Cross_list[Kcross_num-1][1], noise_mean = 0, noise_std_max = 0)  # {103620,853}
    # 数据集划分
    test_size = int(0.5 * len(testset))
    vali_size = len(testset) - test_size
    validation_dataset, test_dataset = torch.utils.data.random_split(testset, [vali_size, test_size])

    validation_data = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)
    test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=workers)
    """
    ====================================
    2、Load model
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create the generator
    net_teacher = Models.Res3D().to(device)
    net_teacher = nn.DataParallel(net_teacher, list(range(ngpu)))
    state_dict = torch.load('./result/checkpoint/1716611673_K1_L300.pth')
    net_teacher.load_state_dict(state_dict)
    print(net_teacher)

    net_student = MobileNet.MLP_Net().to(device)
    print(net_student)


    """
    ====================================
    3、Initial set
    """
    # Loss Functions and Optimizers
    loss_function = nn.CrossEntropyLoss()
    criterion = nn.KLDivLoss()  # KL散度
    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(net_student.parameters(), lr=lr,weight_decay=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    cuda =  True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    savetime = int(time.time())
    save_path = './result/checkpoint/'+ str(savetime) + '_Student' + str(Kcross_num) +'_L'+str(length_set)+ '.pth'
    best_acc = 0.0
    Accuracy = []



"""
====================================
3、训练
"""

for epoch in range(num_epochs):
    ########################################## train ###############################################
    net_student.train()  # 训练过程中开启 Dropout
    running_loss = 0.0  # 每个 epoch 都会对 running_loss  清零
    time_start = time.perf_counter()

    for step, data in enumerate(train_data, start=0):  # 遍历训练集，step从0开始计算
        sensor, labels = data  # 获取训练集的图像和标签
        labels = labels.view(labels.size(0))
        sensor = sensor.to(device, dtype=torch.float)  # 转换到设备  #size (10, 300, 768)
        labels = labels.to(device, dtype=torch.long)




        output_teacher  = net_teacher(sensor)  # 正向传播
        output_student = net_student(sensor)  # 正向传播

        # 计算学生模型预测结果和教师模型预测结果之间的KL散度
        loss_soft = criterion(output_student, output_teacher)
        # 计算学生模型和真实标签之间的交叉熵损失函数值
        loss_hard = loss_function(output_student, labels)
        loss = 0.9 * loss_soft + 0.1 * loss_hard #


        optimizer.zero_grad()  # 清除历史梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数
        running_loss += loss.item()

        # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / len(train_data)  # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

    print()
    print('%f s' % (time.perf_counter() - time_start))

    ########################################### validate ###########################################
    net_student.eval()  # 验证过程中关闭 Dropout
    acc = 0.0
    with torch.no_grad():
        for val_data in validation_data:
            val_sensor, val_labels = val_data
            val_labels = val_labels.view(val_labels.size(0))
            val_sensor = val_sensor.to(device, dtype=torch.float)  # 转换统一数据格式
            outputs = net_student(val_sensor)
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()

        val_num = validation_dataset.__len__()  # 测试集总数

        val_accurate = acc / val_num
        Accuracy.append(val_accurate)

        # 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net_student.state_dict(), save_path)

        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))
    scheduler.step()

print('Finished Training')

