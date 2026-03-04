'''
Spatiotemporal Touch Perception network
Res3D
==============
**Author**: `zhibin Li`__
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import pandas as pd


class ResBlock(nn.Module):
    def __init__(self, in_channel,out_channel, spatial_stride=1,temporal_stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel,kernel_size=(3,3,3),stride=(temporal_stride,spatial_stride,spatial_stride),padding=(1,1,1))
        self.conv2 = nn.Conv3d(out_channel, out_channel,kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample=nn.Sequential(nn.Conv3d(in_channel, out_channel,kernel_size=1,stride=(temporal_stride,spatial_stride,spatial_stride),bias=False),
                                           nn.BatchNorm3d(out_channel))
        else:
            self.down_sample=None

    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2(x_branch)
        x_branch = self.bn2(x_branch)
        if self.down_sample is not None:
            x=self.down_sample(x)
        return self.relu(x_branch+x)

class Res3D(nn.Module):
    # Input size: 8x224x224
    def __init__(self, num_class = 11):
        super(Res3D, self).__init__()

        self.conv11 = nn.Conv3d(1,32,kernel_size=(10,10,10),stride=(1,1,1),padding=(10,0,0))            # P0
        self.conv12 = nn.Conv3d(32, 64, kernel_size=(10, 10, 10), stride=(1, 1, 1), padding=(0, 0, 0)) # 20

        #(10,64,50, 8,8)
        self.conv2  = nn.Sequential(ResBlock(64,64,spatial_stride=1,temporal_stride=1),
                                 ResBlock(64, 64))
        self.conv3 = nn.Sequential(ResBlock(64,128,spatial_stride=2,temporal_stride=1),
                                 ResBlock(128, 128))
        self.conv4 = nn.Sequential(ResBlock(128, 256, spatial_stride=4,temporal_stride=1),
                                   ResBlock(256, 256))
        self.conv5 = nn.Sequential(ResBlock(256, 512, spatial_stride=4,temporal_stride=1),
                                   ResBlock(512, 512))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear=nn.Linear(512,num_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x_3d):
        x = x_3d.reshape([x_3d.size()[0], 1, x_3d.size()[1], 32, 24])  # torch.Size([10, 100, 1, 32, 24])
        x = self.conv11(x)   #
        x = self.conv12(x)  # [10, 64, 50, 8, 8]
        x=self.conv2(x)   #[10, 64, 50, 8, 8]
        x=self.conv3(x)   #[10, 128, 25, 8, 8]
        x=self.conv4(x)   #[10, 256, 13, 8, 8]
        x = self.conv5(x) #[10, 512, 7, 4, 4]
        x = self.avg_pool(x)
        #x = torch.mean(x, dim=2)
        x = self.linear(x.view(x.size(0),-1)) #[10,10]
        x = self.softmax(x)

        return x




class C3D(nn.Module):

    def __init__(self):
        super(C3D, self).__init__()


        self.conv1 = nn.Conv3d(1, 64, kernel_size=(10, 2, 2), )
        self.pool1 = nn.MaxPool3d(kernel_size=(10, 2, 2), stride=(15, 1, 1))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(10, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(15, 1, 1))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(4, 3, 3), padding=(0, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(4, 3, 3), padding=(0, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(15, 2, 2))
        #

        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)


        #
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)


        h = h.view(-1, 256)
        h = self.relu(self.fc4(h))
        h = self.dropout(h)
        h = self.relu(self.fc5(h))
        h = self.dropout(h)
        #
        h = self.fc6(h)
        h = self.softmax(h)

        return h


class C2D(nn.Module):

    def __init__(self):
        super(C2D, self).__init__()


        self.conv1 = nn.Conv2d(1, 64, kernel_size=(10, 2), )
        self.pool1 = nn.MaxPool2d(kernel_size=(15, 2), stride=(15, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(10, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(15, 4), stride=(15, 4))
        #
        self.conv3a = nn.Conv2d(128, 256, kernel_size=(4, 3), padding=(0, 1))
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(4, 3), padding=(0, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(10, 4), stride=(10, 4))

        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)           #[10, 64, 499, 14]

        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        #
        #
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        #
        #
        h = h.view(-1, 256)
        h = self.relu(self.fc4(h))
        h = self.dropout(h)
        h = self.relu(self.fc5(h))
        h = self.dropout(h)
        #
        h = self.fc6(h)
        h = self.softmax(h)

        return h


class LeNetVariant(nn.Module):
    def __init__(self):
        super(LeNetVariant, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(nn.Linear(32 * 6 * 4, 512),
                                        nn.Linear(512, 256))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 6 * 4)
        x = self.classifier(x)
        return x



class CNNLSTM(nn.Module):
    def __init__(self, num_classes=11):
        super(CNNLSTM, self).__init__()
        self.cnn = LeNetVariant()
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=4,
                            batch_first=True)
        self.fc1 = nn.Linear(128, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_3d):
        x_3d = x_3d.reshape([x_3d.size()[0], x_3d.size()[1],1 ,32,24])    #torch.Size([10, 100, 1, 32, 24])
        cnn_output_list = list()
        for t in range(x_3d.size(1)):
            cnn_output_list.append(self.cnn(x_3d[:, t, :, :, :]))
        x = torch.stack(tuple(cnn_output_list), dim=1)
        out, hidden = self.lstm(x)
        print(out.size())
        x = out[:, -1, :]
        x = F.relu(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x






if __name__ == '__main__':
    net = Res3D() #CNNLSTM()
    print(net)
    input1 = torch.rand(10, 500, 768)
    #input1 = torch.rand(10, 1, 32,24)

    out = net(input1)
    print(out.size())

    # print(net)
    # #
    # input1 = torch.rand(10, 1 ,768)  #
    # #input1 = torch.rand(10, 1, 1, 32, 24)  #
    # # #
    # out1 = net(input1)
    # print(out1.size())
