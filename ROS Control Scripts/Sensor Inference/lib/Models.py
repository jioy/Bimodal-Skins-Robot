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
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis



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



class MobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(MobileNetV2, self).__init__()
        # Load the pre-trained MobileNetV2 model
        self.model = models.mobilenet_v2(pretrained=True)
        # Modify the first convolutional layer to accept single channel input
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Modify the classifier to output the number of classes
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 32, 24)
        return self.model(x)


# Define a custom regression network based on ShuffleNet
class ShuffleNet(nn.Module):
    def __init__(self):
        super(ShuffleNet, self).__init__()

        # Load the pre-trained ShuffleNet model
        self.shufflenet = models.shufflenet_v2_x1_0(weights=None)  # Do not use pre-trained weights

        # Adjust the first convolutional layer to accept single-channel input (originally designed for 3-channel RGB input)
        self.shufflenet.conv1 = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Modify the final fully connected layer to output 3 values (for regression task)
        self.shufflenet.fc = nn.Linear(self.shufflenet.fc.in_features, 3)

    def forward(self, x):
        # Add a channel dimension to the input, resulting in (batch_size, 1, 768)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, 768)

        # Reshape the input to a 2D spatial form (batch_size, 1, height, width)
        # Example: reshape to (batch_size, 1, 32, 24), where 32*24 = 768
        x = x.view(x.size(0), 1, 32, 24)  # Shape: (batch_size, 1, 32, 24)

        # Pass through the ShuffleNet model
        x = self.shufflenet(x)
        return x



class LSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_layers=2, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out, _ = self.lstm(x) # LSTM层
        out = self.fc(out[:, -1, :]) # 全连接层
        return out


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super(MobileViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        # Apply layer normalization and attention
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + x  # Residual connection

        # Apply layer normalization and MLP
        x = self.norm2(x)
        x = self.mlp(x) + x  # Residual connection
        return x


class MobileViT(nn.Module):
    def __init__(self, input_dim=768, output_dim=3, depth=2, heads=4, mlp_dim=512):
        super(MobileViT, self).__init__()

        # MobileViT blocks
        self.mobilevit_block = MobileViTBlock(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim)

        # MLP Head for regression
        self.regressor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # Forward pass through MobileViT block
        x = self.mobilevit_block(x)

        # Average pooling across sequence length (assume x shape: [batch, seq_len, dim])
        x = torch.mean(x, dim=1)  # Average pooling over the sequence length dimension

        # Pass through the MLP regression head
        x = self.regressor(x)
        return x






# Define a custom regression network based on ResNet-18
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        # Load the pre-trained ResNet18 model
        self.resnet18 = models.resnet18(weights=None)  # Do not use pre-trained weights

        # Adjust the first convolutional layer to accept single-channel input (originally designed for 3-channel RGB input)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Modify the final fully connected layer to output 3 values (for regression task)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 3)

    def forward(self, x):
        # Adjust input dimensions to be compatible with ResNet (batch_size, channels, height, width)
        x = x.unsqueeze(1)  # Add a channel dimension, resulting in (batch_size, 1, 768)
        x = self.resnet18(x)
        return x


if __name__ == '__main__':
    # Phy = PhysicalModel()

    ##============================= 一、 TransformerEncoderModel
    net = TransformerEncoderModel(input_dim=768, num_heads=4, hidden_dim=256, output_dim=3, num_layers=1,
                                         dropout_rate=0.1)
