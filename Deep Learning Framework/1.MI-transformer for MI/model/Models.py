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


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_layers=1, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out, _ = self.lstm(x) # LSTM层
        out = self.fc(out[:, -1, :]) # 全连接层
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_layers=1, output_size=3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)


    def forward(self, x):
        out, _ = self.lstm(x) # LSTM层
        out = self.fc(out[:, -1, :]) # 全连接层
        return out


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
    def __init__(self, num_classes=3):
        super(CNNLSTM, self).__init__()
        self.cnn = LeNetVariant()
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2,
                            batch_first=True)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        x_3d = x_3d.reshape([x_3d.size()[0], x_3d.size()[1],1 ,32,24])
        cnn_output_list = list()
        for t in range(x_3d.size(1)):
            cnn_output_list.append(self.cnn(x_3d[:, t, :, :, :]))
        x = torch.stack(tuple(cnn_output_list), dim=1)
        out, hidden = self.lstm(x)
        x = out[:, -1, :]
        x = F.relu(x)
        x = self.fc1(x)
        return x



class MLP_Net(nn.Module):

    def __init__(self):
        super(MLP_Net, self).__init__()

        # 全连接分类器--融合分类器
        self.Sequential = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),  # 1024 或 2048
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=3)

        )

        self.instance_norm = nn.InstanceNorm2d(num_features=1)


    def forward(self, sensor):
        out = self.instance_norm(sensor)
        out = self.Sequential(out)
        out = out[:, -1, :]
        return out


class PhysicalModel():     #USB 400Hz 刷新率
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_data()

    def init_data(self):
        angel_left_F = np.asarray(pd.read_csv("./numpy_lib/left_front1.csv", header=None, index_col=None))
        self.Left_cos = np.cos(angel_left_F * np.pi / 180)
        self.Left_sin = np.sin(angel_left_F * np.pi / 180)
        self.Left_cos = torch.tensor(self.Left_cos,device=self.device)
        self.Left_sin = torch.tensor(self.Left_sin,device=self.device)


        angel_right_F = np.asarray(pd.read_csv("./numpy_lib/right_front1.csv", header=None, index_col=None))
        self.Right_cos = np.cos(angel_right_F * np.pi / 180)
        self.Right_sin = np.sin(angel_right_F * np.pi / 180)
        self.Right_cos = torch.tensor(self.Right_cos,device=self.device)
        self.Right_sin = torch.tensor(self.Right_sin,device=self.device)

        angel_back_F = np.asarray(pd.read_csv("./numpy_lib/behind.csv", header=None, index_col=None))
        self.Back_cos = np.cos(angel_back_F * np.pi / 180)
        self.Back_sin = np.sin(angel_back_F * np.pi / 180)
        self.Back_cos = torch.tensor(self.Back_cos,device=self.device)
        self.Back_sin = torch.tensor(self.Back_sin,device=self.device)


    def frame_plot(self,sensor_data):
        self.sensor_data = sensor_data.numpy()
        sensor1_deta = self.sensor_data[:,:, 0:256]
        sensor2_deta = self.sensor_data[:,:, 256:512]
        sensor3_deta = self.sensor_data[:,:, 512:768]
        Real_Fx = self.sensor_data[:,:, 769]
        Real_Fz = self.sensor_data[:,:, 770]
        Real_Ft = self.sensor_data[:,:, 771]

        # 传感器1
        MAX_DEL = 0

        sensor1_array = sensor1_deta.reshape(sensor1_deta.shape[0],sensor1_deta.shape[1], 16, 16)
        sensor1_array[sensor1_array < MAX_DEL] = MAX_DEL
        # 传感器2
        sensor2_array = sensor2_deta.reshape(sensor2_deta.shape[0],sensor2_deta.shape[1], 16, 16)
        sensor2_array[sensor2_array < MAX_DEL] = MAX_DEL
        # 传感器3
        sensor3_array = sensor3_deta.reshape(sensor3_deta.shape[0],sensor3_deta.shape[1], 16, 16)
        sensor3_array[sensor3_array < MAX_DEL] = MAX_DEL

        sensor3a_array = sensor3_array[:,:, 0:8, -5:]
        sensor3b_array = sensor3_array[:,:, 8:16, -5:]

        # 位置变换#
        sensor1_array = np.rot90(sensor1_array, k=1, axes=(2, 3))  # 旋转90度
        sensor1_array = np.flip(sensor1_array, axis=3)  # 上下反转
        sensor2_array = np.rot90(sensor2_array, k=1, axes=(2, 3))  # 旋转90度

        # 力学求解：
        sensor1_Fx = sensor1_array * self.Left_cos
        sensor2_Fx = sensor2_array * self.Right_cos
        sensor3a_Fx = sensor3a_array * self.Back_cos
        sensor3b_Fx = sensor3b_array * self.Back_cos

        sensor1_Fz = sensor1_array * self.Left_sin
        sensor2_Fz = sensor2_array * self.Right_sin
        sensor3a_Fz = sensor3a_array * self.Back_sin
        sensor3b_Fz = sensor3b_array * self.Back_sin

        Fx_l = np.mean(sensor1_Fx, axis=(2, 3)) + np.mean(sensor3a_Fx, axis=(2, 3))
        Fx_r = np.mean(sensor2_Fx, axis=(2, 3)) + np.mean(sensor3b_Fx, axis=(2, 3))
        Fz_l = np.mean(sensor1_Fz, axis=(2, 3)) + np.mean(sensor3a_Fz, axis=(2, 3))
        Fz_r = np.mean(sensor2_Fz, axis=(2, 3)) + np.mean(sensor3b_Fz, axis=(2, 3))

        Fx = (Fz_l + Fz_r) * 15
        Fz = (Fz_l + Fz_r - 2) * 6
        T = (Fx_l - Fx_r) * 1.2


        # Fx = Fx + 65.04
        # Fz = Fz - 67.7
        #
        print(self.sensor_data.shape)
        print(Fx.shape)
        plt.figure(1)
        plt.plot(self.sensor_data[0,:, 0], Fx[0,:])  # 画线
        plt.plot(self.sensor_data[0,:, 0], Real_Fx[0,:], color='red')
        plt.figure(2)
        plt.plot(self.sensor_data[0,:, 0], Fz[0,:])  # 画线
        plt.plot(self.sensor_data[0,:, 0], Real_Fz[0,:], color='red')
        plt.figure(3)
        plt.plot(self.sensor_data[0,:, 0], T[0,:])  # 画线
        plt.plot(self.sensor_data[0,:, 0], Real_Ft[0,:], color='red')
        plt.show()  # 显示图形

        #(1, 5459, 3)
        outdata_physical = np.concatenate((Fx.reshape(Fx.shape[0],Fx.shape[1],1),
                                           Fz.reshape(Fz.shape[0],Fz.shape[1],1),
                                           T.reshape(T.shape[0],T.shape[1],1)), axis=2)
        #

        outdata_physical = torch.Tensor(outdata_physical)
        print(outdata_physical.size())


    def frame_caculate(self,sensor_data):
        self.sensor_data = sensor_data
        sensor1_deta = self.sensor_data[:,:, 0:256]
        sensor2_deta = self.sensor_data[:,:, 256:512]
        sensor3_deta = self.sensor_data[:,:, 512:768]

        # 传感器1
        MAX_DEL = 0

        sensor1_array = sensor1_deta.reshape(sensor1_deta.shape[0],sensor1_deta.shape[1], 16, 16)
        sensor1_array[sensor1_array < MAX_DEL] = MAX_DEL
        # 传感器2
        sensor2_array = sensor2_deta.reshape(sensor2_deta.shape[0],sensor2_deta.shape[1], 16, 16)
        sensor2_array[sensor2_array < MAX_DEL] = MAX_DEL
        # 传感器3
        sensor3_array = sensor3_deta.reshape(sensor3_deta.shape[0],sensor3_deta.shape[1], 16, 16)
        sensor3_array[sensor3_array < MAX_DEL] = MAX_DEL

        sensor3a_array = sensor3_array[:,:, 0:8, -5:]
        sensor3b_array = sensor3_array[:,:, 8:16, -5:]

        # 位置变换#
        sensor1_array = torch.rot90(sensor1_array, k=1, dims=(2, 3))  # 旋转90度
        sensor1_array = torch.flip(sensor1_array, dims=[3])  # 上下反转
        sensor2_array = torch.rot90(sensor2_array, k=1, dims=(2, 3))  # 旋转90度

        # 力学求解：
        sensor1_Fx = sensor1_array * self.Left_cos.to(sensor1_array.device)
        sensor2_Fx = sensor2_array * self.Right_cos.to(sensor2_array.device)
        sensor3a_Fx = sensor3a_array * self.Back_cos.to(sensor3a_array.device)
        sensor3b_Fx = sensor3b_array * self.Back_cos.to(sensor3b_array.device)

        sensor1_Fz = sensor1_array * self.Left_sin.to(sensor1_array.device)
        sensor2_Fz = sensor2_array * self.Right_sin.to(sensor2_array.device)
        sensor3a_Fz = sensor3a_array * self.Back_sin.to(sensor3a_array.device)
        sensor3b_Fz = sensor3b_array * self.Back_sin.to(sensor3b_array.device)

        Fx_l = torch.mean(sensor1_Fx, axis=(2, 3)) + torch.mean(sensor3a_Fx, axis=(2, 3))
        Fx_r = torch.mean(sensor2_Fx, axis=(2, 3)) + torch.mean(sensor3b_Fx, axis=(2, 3))
        Fz_l = torch.mean(sensor1_Fz, axis=(2, 3)) + torch.mean(sensor3a_Fz, axis=(2, 3))
        Fz_r = torch.mean(sensor2_Fz, axis=(2, 3)) + torch.mean(sensor3b_Fz, axis=(2, 3))

        Fz = (Fz_l + Fz_r) * 5
        T = (Fx_l - Fx_r)  *5 *0.2
        Fx = (Fx_l + Fx_r) * 5
        #Fx = (Fz_l + Fz_r - 2) * 6

        #(1, 5459, 3)
        outdata_physical = torch.cat((Fx,Fz,T), dim=1)   #[10, 3]
        return outdata_physical


# 定义倒残差模块（Inverted Residual Block）
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise convolution (expansion)
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise convolution
        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 1x1 pointwise convolution (projection)
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 定义 MobileNetV2 主网络
class MobileNetV2(nn.Module):
    def __init__(self, num_ch=3, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        # 每一层的配置参数：输出通道数，卷积步长，扩展因子
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 1, 2],
            [6, 32, 1, 2],
            [6, 64, 1, 2],
            [6, 96, 1, 1],
            [6, 160, 1, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        last_channel = 32

        # 调整初始卷积层以适应单通道输入（通道数从 3 改为 1）
        self.features = [nn.Conv2d(1, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(input_channel),
                         nn.ReLU6(inplace=True)]

        # 添加倒残差模块
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 添加最后一层卷积
        last_conv = int(last_channel * max(1.0, width_mult))
        self.features.append(nn.Conv2d(input_channel, last_conv, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.append(nn.BatchNorm2d(last_conv))
        self.features.append(nn.ReLU6(inplace=True))

        # 将特征层转换为 nn.Sequential
        self.features = nn.Sequential(*self.features)

        # 分类器
        self.pre = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_conv, num_ch),
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, 1, 32, 24)
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.pre(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)




class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim=768, num_heads=4, hidden_dim=256, output_dim=3, num_layers=1, dropout_rate=0.2):
        super(TransformerEncoderModel, self).__init__()

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear layer to map to the desired output dimension
        self.Sequential = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),  # 1024 或 2048
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=output_dim)

        )



    def forward(self, x):
        # x shape: (seq_len, batch_size, input_dim)
        x = self.transformer_encoder(x)
        # Since the input is (1, 768), we assume seq_len = 1, so we'll squeeze the seq_len dimension
        x = x.squeeze(0)  # x shape now: (batch_size, input_dim)
        x = self.Sequential(x)  # Final linear layer to map to output_dim
        x = x.contiguous().view(-1, 3)
        return x





if __name__ == '__main__':
    # Phy = PhysicalModel()
    # y1 = Phy.frame_caculate(input1)

    # net = MLP_Net()
    # input1 = torch.rand(10, 1, 768)
    # out = net(input1)
    # print(out.size())

    # 创建 MobileNetV2 模型实例
    # net = MobileNetV2(num_ch=3)
    # input1 = torch.rand(10, 1,768)
    # out = net(input1)
    # print(out.size())

    net = TransformerEncoderModel(input_dim=768, num_heads=4, hidden_dim=256, output_dim=3, num_layers=1,
                                         dropout_rate=0.1)
    print(net)
    input1 = torch.rand(10, 1 ,768)  #
    out1 = net(input1)
    print(out1.size())

    # 计算模型参数的总大小
    total_size = sum(torch.numel(param) for param in net.parameters())
    # 将参数总大小转换为M单位
    total_size_in_millions = total_size / 1_000_000
    print(f"Total model size: {total_size_in_millions:.2f} M parameters")
