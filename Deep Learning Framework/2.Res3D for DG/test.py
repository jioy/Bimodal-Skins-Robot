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
import model.Models as Models #加载数据集
import random
import math
import argparse
from torchmetrics.regression import R2Score #求R2
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
import argparse
from sklearn.metrics import confusion_matrix
import csv


from sklearn.metrics import roc_curve, auc
from itertools import cycle



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

def write_data(filenname, data):
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
    batch_size = 20  # 10

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 2

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
    Cross_list = [] #6折交叉验证划分，Kcross_num为第几折用于测试
    kf = KFold(n_splits=6, shuffle=True, random_state=0)
    for cross_train, cross_test in kf.split(list(range(0, 12))):
        Cross_list.append([list(cross_train + 1), list(cross_test+1)])


    # Create the test_data
    test_path = r"../Alldata"
    Testset_all = dataload.MyDataset(test_path,time_lenth = length_set,
                                 k_num = Cross_list[Kcross_num-1][1], noise_mean = 0, noise_std_max = 0)  # {103620,853}
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
    net = nn.DataParallel(net, list(range(1)))
    print(net)

    state_dict = torch.load('./result/checkpoint/1718260974_K1_L100.pth')
    net.load_state_dict(state_dict)



    """
    ====================================
    3、Test
    """
    net.eval()
    acc = 0.0
    y_test = []
    y_pred = []

    # Assuming num_classes is the number of classes in your classification task
    num_classes = 11  # Replace with your actual number of classes
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for test_onedata in test_data:
            test_sensor, test_labels = test_onedata
            test_labels = test_labels.view(test_labels.size(0))
            test_sensor = test_sensor.to(device, dtype=torch.float)
            test_labels = test_labels.to(device, dtype=torch.long)

            outputs = net(test_sensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
            predict_y = torch.max(outputs, dim=1)[1]  # Predicted labels

            # Store true labels and predicted probabilities
            all_labels.extend(test_labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

            y_test.extend(list(test_labels.cpu().numpy()))
            y_pred.extend(list(predict_y.cpu().numpy()))

            acc += (predict_y == test_labels).sum().item()

        val_num = len(test_data.dataset)  # Get the total number of samples
        val_accurate = acc / val_num
        print('testdata_accuracy: %.3f \n' % val_accurate)

    # Convert lists to numpy arrays for ROC calculation
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange',
                    'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta'])

    plt.figure()
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    # Add thinner border lines
    plt.plot([0, 1], [0, 1], 'k--', lw=1)  # Thinner diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Increase font size
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)

    # Set ticks font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('./result/fig/roc_curve.svg', format='svg', dpi=600)
    plt.show()




    # === 混淆矩阵：真实值与预测值的对比 ===


    # y_test = np.array(y_test)
    # y_pred = np.array(y_pred)
    # y_test = y_test.reshape(-1, 1)
    # y_pred = y_pred.reshape(-1, 1)
    # print(y_test.shape)
    # print(y_pred.shape)
    #
    # con_mat = confusion_matrix(y_test, y_pred)
    #
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    # con_mat_norm = np.around(con_mat_norm, decimals=2)
    # print(con_mat_norm)
    # write_data('./result/fig/Confusion.csv', con_mat_norm)

