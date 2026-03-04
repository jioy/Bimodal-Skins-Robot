# -*- coding: utf-8 -*-
"""
Load data
==============
**Author**: `zhibin Li`
"""
# Create the dataset  Load sensor data
# 触觉传感器数据读取
####################################################################
import natsort  # 第三方排序库
from PIL import Image
from torchvision.transforms import ToPILImage
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import torchvision.transforms as transforms
import os
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import random
from scipy.signal import savgol_filter as sgolay
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision import transforms
import pyqtgraph as pg
import math
import pickle
import os
import sys
if os.getcwd() != sys.path[0]:  #添加当前运行目录
    os.chdir(sys.path[0])





class MyDataset(Dataset):  #Dataset
    def __init__(self, path_dir = '../Processed data/', transform=None, time_lenth = 1, k_num = [1], noise_mean = 0,noise_std_max = 1):
        self.time_lenth = time_lenth   #时间序列长度
        print('time_lenth',self.time_lenth)
        self.path_dir = path_dir  # file path
        self.transform = transform  # transform
        self.participants = os.listdir(self.path_dir) # file path list
        self.participants.sort() # 排序
        self.Sensor_data = []  #sensor data list  [frame num, 853] (853: the data frame size)
        self.Sensor_Label = []
        # data path 和 Label
        #################################
        self.Sensor_list = []
        self.Label_list = []
        for participants_name in k_num:  #data
            file_dir = self.path_dir + '/P' + str(participants_name)  # sensor data patch

            for filename in range(1,12):

                data_dir = file_dir + '/Gesture (' + str(filename) + ').csv'  # sensor data patch

                #读取数据
                readnumpy = np.array(pd.read_csv(data_dir))

                for i in range(len(readnumpy)-self.time_lenth): #时间序列分割
                    Sensor_data = readnumpy[i:i + time_lenth, 1:]
                    Label_data = [filename-1]

                    if(noise_std_max > 0): # add noise
                        noise_std = random.random()*noise_std_max
                        noise = np.random.normal(noise_mean, noise_std, [Sensor_data.shape[0], Sensor_data.shape[1]])
                        Sensor_data = Sensor_data + noise

                    self.Label_list.append(Label_data)
                    self.Sensor_list.append(Sensor_data)

        self.Sensor_array = np.array(self.Sensor_list)
        self.Label_array = np.array(self.Label_list)

        self.len = len(self.Label_array)  #sensor length


        self.Sensor_data = torch.Tensor(self.Sensor_array)
        self.Sensor_label = torch.Tensor(self.Label_array)

        self.Sensor_data = F.normalize(self.Sensor_data, dim=2)



    def __len__(self):  # dataset length
        return self.len

    def alldata(self):  # index get
        sensor_data = self.Sensor_data.numpy()
        return sensor_data

    def __getitem__(self, index):  # index get
        sensor_data = self.Sensor_data[index]
        sensor_label = self.Sensor_label[index]
        return sensor_data,sensor_label



if __name__ == '__main__':
    Alldata_list = list(range(1, 13))
    Data = MyDataset("../../Alldata",time_lenth = 1, k_num = Alldata_list, noise_mean = 0, noise_std_max = 1)
    print(Data.__len__())
    readdata,label = Data.__getitem__(1000) # get the 34th sample
    print(readdata)
    print(label)

    # print(sensor.size()) #torch.Size([5000, 8, 8])
    # print(label.size())

    # sensor = Data[:, 1: 769]  # 压力数据
    # labels = Data[:, 772 -3: 772]  # 车速数据
    #
    # print(sensor)
    # print(labels)