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
from fvcore.nn import FlopCountAnalysis
from torchvision.models import resnet18
from torchvision import models

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


def gn(c, g=4):
    # GroupNorm with `g` groups
    return nn.GroupNorm(g, c)


class ResBlock_lite(nn.Module):
    def __init__(self, in_channel, out_channel, spatial_stride=1, temporal_stride=1):
        super(ResBlock_lite, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3),
                               stride=(temporal_stride, spatial_stride, spatial_stride), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.norm1 = gn(out_channel)
        self.norm2 = gn(out_channel)
        self.relu = nn.ReLU()

        # Downsample layer for residual connection if required
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=1,
                          stride=(temporal_stride, spatial_stride, spatial_stride), bias=False),
                gn(out_channel)
            )
        else:
            self.down_sample = None

    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.norm1(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2(x_branch)
        x_branch = self.norm2(x_branch)

        # Apply the residual connection
        if self.down_sample is not None:
            x = self.down_sample(x)

        return self.relu(x_branch + x)


class Res3D_lite(nn.Module):   # Accuracy ~99.5%
    def __init__(self, num_class=11):
        super(Res3D_lite, self).__init__()

        # Reduced kernel size and number of filters to make it more lightweight
        self.conv11 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv12 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        # Sequential ResBlocks with reduced feature sizes
        self.conv2 = nn.Sequential(
            ResBlock_lite(32, 32, spatial_stride=1, temporal_stride=1),
            ResBlock_lite(32, 32)
        )
        self.conv3 = nn.Sequential(
            ResBlock_lite(32, 64, spatial_stride=2, temporal_stride=1),
            ResBlock_lite(64, 64)
        )
        self.conv4 = nn.Sequential(
            ResBlock_lite(64, 128, spatial_stride=2, temporal_stride=1),
            ResBlock_lite(128, 128)
        )
        self.conv5 = nn.Sequential(
            ResBlock_lite(128, 256, spatial_stride=2, temporal_stride=1),
            ResBlock_lite(256, 256)
        )

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(256, num_class)

    def forward(self, x_3d):
        # Adjust input dimensions to (B, 1, T, H, W)
        x = x_3d.reshape([x_3d.size()[0], 1, x_3d.size()[1], 32, 24])
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv2(x)  # Reduced features
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Adaptive pooling
        x = self.avg_pool(x)
        x = self.linear(x.view(x.size(0), -1))  # Flatten the output for classification

        return x




class ResNet18(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet18, self).__init__()
        # 使用ResNet18的预训练模型
        self.resnet = resnet18(pretrained=False)

        # 调整输入通道数为100
        self.resnet.conv1 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改全连接层的输出大小为 num_classes
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = x.reshape([x.size()[0], x.size()[1], 32, 24])  # torch.Size([10, 100, 1, 32, 24])
        return self.resnet(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet34, self).__init__()
        # 使用ResNet18的预训练模型
        self.resnet = models.resnet34(pretrained=False)

        # 调整输入通道数为100
        self.resnet.conv1 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改全连接层的输出大小为 num_classes
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = x.reshape([x.size()[0], x.size()[1], 32, 24])  # torch.Size([10, 100, 1, 32, 24])
        return self.resnet(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet50, self).__init__()
        # 使用ResNet18的预训练模型
        self.resnet = models.resnet50(pretrained=False)

        # 调整输入通道数为100
        self.resnet.conv1 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改全连接层的输出大小为 num_classes
        self.resnet.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        x = x.reshape([x.size()[0], x.size()[1], 32, 24])  # torch.Size([10, 100, 1, 32, 24])
        return self.resnet(x)


class C2D(nn.Module):
    def __init__(self, num_classes=11):
        super(C2D, self).__init__()

        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化层

        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 3, 512)  # 计算输入到全连接层的大小
        self.fc2 = nn.Linear(512, num_classes)  # 输出层，分类为 num_classes

        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 加入dropout防止过拟合

    def forward(self, x):
        # 输入的形状为 (batch_size, 100, 32, 24)
        x = x.reshape([x.size()[0], x.size()[1], 32, 24])  # torch.Size([10, 100, 1, 32, 24])
        # 卷积层和池化层
        x = self.pool(self.relu(self.conv1(x)))  # 输出形状: (batch_size, 64, 16, 12)
        x = self.pool(self.relu(self.conv2(x)))  # 输出形状: (batch_size, 128, 8, 6)
        x = self.pool(self.relu(self.conv3(x)))  # 输出形状: (batch_size, 256, 4, 3)

        # 展平张量
        x = x.view(-1, 256 * 4 * 3)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # dropout层
        x = self.fc2(x)  # 输出层

        return x







class C3D(nn.Module):
    def __init__(self, num_classes=11):
        super(C3D, self).__init__()

        # 3D convolution and pooling layers
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Adaptive average pooling layer to dynamically adjust to (1, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected layers
        self.fc4 = nn.Linear(256, 128)  # The input size is now fixed to 256 after adaptive pooling
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through the network
        x = x.reshape([x.size()[0], 1, x.size()[1], 32, 24])  # torch.Size([10, 100, 1, 32, 24])
        h = self.relu(self.conv1(x))  # Conv1 + ReLU
        h = self.pool1(h)  # Pool1

        h = self.relu(self.conv2(h))  # Conv2 + ReLU
        h = self.pool2(h)  # Pool2

        h = self.relu(self.conv3a(h))  # Conv3a + ReLU
        h = self.relu(self.conv3b(h))  # Conv3b + ReLU
        h = self.pool3(h)  # Pool3

        # Apply adaptive average pooling to get a fixed-size output of (1, 1, 1)
        h = self.avg_pool(h)

        # Flatten the feature map into a vector for the fully connected layers
        h = h.view(h.size(0), -1)  # Shape becomes (batch_size, 256)

        # Fully connected layers
        h = self.relu(self.fc4(h))  # FC4 + ReLU
        h = self.dropout(h)  # Dropout after FC4
        h = self.relu(self.fc5(h))  # FC5 + ReLU
        h = self.dropout(h)  # Dropout after FC5

        # Output layer
        h = self.fc6(h)  # FC6 (final output)
        h = self.softmax(h)  # Softmax for classification

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



class MobileNetV2(nn.Module):
    def __init__(self, num_classes=11):
        super(MobileNetV2, self).__init__()
        # Load the pre-trained MobileNetV2 model
        self.model = models.mobilenet_v2(pretrained=True)
        # Modify the first convolutional layer to accept single channel input
        self.model.features[0][0] = nn.Conv2d(100, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Modify the classifier to output the number of classes
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        x = x.reshape([x.size()[0], x.size()[1], 32, 24])
        return self.model(x)







if __name__ == '__main__':
    net = C2D()
    print(net)
    input1 = torch.rand(2, 100, 768)

    #input1 = torch.rand(10, 1, 32,24)

    out = net(input1)
    print(out.size())
    
    
    
    # Perform FLOPs analysis
    flops = FlopCountAnalysis(net, input1)
    print(f"FLOPs: {flops.total()/ 1_000_000/2} M")

    # 计算模型参数的总大小
    total_size = sum(torch.numel(param) for param in net.parameters())
    # 将参数总大小转换为M单位
    total_size_in_millions = total_size / 1_000_000
    print(f"Total model size: {total_size_in_millions:.2f} M parameters")


    

    # print(net)
    # #
    # input1 = torch.rand(10, 1 ,768)  #
    # #input1 = torch.rand(10, 1, 1, 32, 24)  #
    # # #
    # out1 = net(input1)
    # print(out1.size())
