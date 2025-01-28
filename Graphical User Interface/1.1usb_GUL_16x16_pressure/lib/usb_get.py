'''
Read Skin sensor data
==============
**Author**: `zhibin Li`
'''


import serial
import time
import struct
import multiprocessing
from multiprocessing import Queue
import os
import numpy as np
import time
import keyboard
import sys
from PIL import Image
import pyqtgraph as pg
import cv2

class USB_Connect:     #USB 400Hz 刷新率

    def __init__(self):
        super().__init__()


        # 读取参数
        f = open('./lib/coefficient.txt')
        coefficient_data = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
        f.close()  # 关

        # ['Load ratio parameter:\n', '3']
        # 参数设置
        self.coefficient = float(coefficient_data[1])
        print('比例参数为：', self.coefficient)


    def Message_decode(self,data_flag,com):

        try:
            self.com = serial.Serial(com, 2000000)
            print('串口连接成功')

        except Exception as e:
            print("---异常---：", e)
            print("---硬件串口异常---：")
            sys.exit(0)



        while True:
            #包头截取
            dd = self.com.read(1)
            if (dd == b'\xbb'):
                if(self.com.read(1) != b'\xbb'):
                    continue
            else: continue

            while True:
                data = self.com.read((256+2)*2)

                if(data[0:2] != b'\xaa\xaa' or data[-2:] != b'\xbb\xbb'): #包头，包尾核对
                    print('erro')
                    break
                #print('ok')

                data_flag.put(data)



    def sendMessage(self, message):
        self.tcp_client.write(message.encode(encoding='utf-8'))


    def closeClient(self):
        self.com.close()
        print("串口已关闭")


    #清空队列
    def clear_Queue(self,q):
        res = []
        while q.qsize() > 0:
            res.append(q.get())



    def init_data(self,data_flag):
        init_data_buffer = [0 for i in range(0,256)]
        data_buffer = {n: [] for n in range(256)}

        for i in range(20):

            udp_data = data_flag.get(True)  # 接收数据

            for i in range(0, 256):
                data_analysis = struct.unpack('<h', udp_data[2 + i * 2: 4 + i * 2])  # 元组  2字节
                data_decode = data_analysis[0] * self.coefficient

                data_buffer[i].append(data_decode)


        for i in range(256):
            init_data_buffer[i] = np.max(data_buffer[i])

        return init_data_buffer



    # 膝关节重建 图像获取
    def get_image(self):
        image_dir = r"./lib/AI/robot2.png"
        self.joint = Image.open(image_dir)  # 打开图片
        self.joint = self.joint.convert("RGB") #变RGB,三通道
        self.joint = self.joint.resize((600, 600))
        self.joint = self.joint.rotate(270)  # 逆时针旋转90

        self.joint_h = self.joint.convert('L')
        self.joint = np.asarray(self.joint)  # 转换为矩阵
        self.joint_h = np.asarray(self.joint_h)  # 转换为矩阵

        self.coord_A = [[53,147],[120, 147],[190, 147],
                        [27,206],[88,206],[153,206],[217,206],
                        [28,264],[88,264],[153,264],[217,264],
                        [25,320],[87,320],[153,320],[220,320],
                        [32, 376], [93, 376], [158, 376], [224, 376],
                        [42, 430], [104, 430], [169, 430], [234, 430],
                        [68, 486], [137, 486], [208, 486],
                        ]

        self.coord_B = [[403, 147], [477, 147], [550, 147],
                        [374, 206], [442, 206], [512, 206], [575, 206],
                        [375, 264], [444, 264], [510, 264], [575, 264],
                        [368, 320], [435, 320], [507, 320], [570, 320],
                        [362, 376], [428, 376], [497, 376], [563, 376],
                        [352, 430], [419, 430], [487, 430], [552, 430],
                        [362, 486], [435, 486], [508, 486],
                        ]

        self.white = np.zeros((600, 600,3))
        self.black = np.zeros((600, 600, 3))
        for w in range(600):
            for h in range(600):
                if (self.joint_h[w,h]==255):
                    self.white[w,h,:] = 0
                    self.black[w,h,:] = 255
                else:
                    self.white[w, h, :] = 1
                    self.black[w, h, :] = 0



    def image_reconstructed(self,data_z,image_construct):
        self.get_image()
        self.value_based = 0;

        while True:

            get_z = data_z.get(True)[0] #接收数据
            self.clear_Queue(data_z)


            #sensor = np.zeros((640, 640))
            sensor = np.ones((600, 600)) * self.value_based
            #36, 198   582,399

            ave_value = 0
            for i in range(16):
                for j in range(16):
                    sensor[(i)*34 + 36,(15-j)*12 + 198] = get_z[i,j] *0.2 + self.value_based
                    ave_value = get_z[i,j]/64/200  + ave_value

            sensor = pg.gaussianFilter(sensor, (15, 15))  # 高斯平滑
            sensor *= 255  # 变换为0-255的灰度值

            # 限制数值范围在0-255之间，并转换为uint8类型
            sensor = np.clip(sensor, 0, 255).astype(np.uint8)
            # 热力图  避免颜色变化
            sensor[0, 0] = 255

            # 热力图
            heat_img = cv2.applyColorMap(sensor, cv2.COLORMAP_CIVIDIS)  # 注意此处的三通道热力图是cv2专有的GBR排列 COLORMAP_JET COLORMAP_CIVIDIS
            heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像

            # 足底图像
            heat_img = (heat_img * self.white) + self.black  # 消除伪影
            plotdata = heat_img

            image_construct.put([plotdata,ave_value])


    def usb_decode(self,data_flag,data_out,data_z,GUI_order):
        data_buffer = [0] * 256
        send_savedata = np.zeros((257))

        z = np.zeros((16, 16))
        Sensor = np.zeros((16, 16))

        send_flag = 0  #装载数据标志位
        strat_time = time.time()  #采样计时

        #初始化，求均值
        init_data = self.init_data(data_flag)



        while True:
            udp_data = data_flag.get(True) #接收数据

            for i in range(0, 256):
                data_analysis = struct.unpack('<h', udp_data[2 + i * 2: 4 + i * 2])  # 元组  2字节
                data_decode = data_analysis[0] * self.coefficient - init_data[i]
                send_savedata[i+1] = data_decode

                if (data_decode < 0):  data_decode = 0
                data_buffer[i] = data_decode

            for i in range(16):
                for j in range(16):
                    z[i , j] = data_buffer[i*16 + j]
                    Sensor[i,j] = data_buffer[i*16 + j]


            data_z.put([z,0])

            # 是否保存数据 数据保存
            if (GUI_order.empty() == False):  # 接收到采样标志后开始采样
                get_flag = GUI_order.get()
                if (get_flag == 'start'):  # 开始采样装入
                    self.clear_Queue(data_out)  # 清空缓存
                    strat_time = time.time()
                    send_flag = 1
                if (get_flag == 'stop'):  # 开始采样装入
                    deta_time = time.time() - strat_time
                    print(deta_time)
                    send_flag = 0

            if (send_flag == 1):
                timestamp = int(time.time())
                send_savedata[0] = timestamp
                data_out.put(send_savedata)








class USB_DataDecode:

    def __init__(self,com):
        super().__init__()

        #self.draw_3d = plot3d.PLOT_3D()
        self.com_num = com

        #多进程
        self.data_flag = Queue() #更新状态
        self.data_out = Queue()  #数据解析结果

        self.data_z = Queue()  # 数据解析结果   //接收解析数据

        self.image_construct = Queue()  #图像重建结果

        self.GUI_order = Queue()  # GUI控制数据解析


        #进程1 接收数据
        self.T = USB_Connect()
        self.thread_getMessage = multiprocessing.Process(target=self.T.Message_decode,args=(self.data_flag,self.com_num))

        #进程2 处理数据
        self.thread_usbdecode = multiprocessing.Process(target=self.T.usb_decode,args=(self.data_flag,self.data_out,self.data_z,self.GUI_order))

        #进程3 图像重建
        self.thread_image_construct = multiprocessing.Process(target=self.T.image_reconstructed, args=(
        self.data_z, self.image_construct))

        #self.thread_key_monitoring = multiprocessing.Process(target=self.key_monitoring, args=())


        #self.eat_process = multiprocessing.Process(target=self.eat, args=(3, "giao"))
        print("主进程ID:", os.getpid())
        self.thread_getMessage.start()
        self.thread_usbdecode.start()
        self.thread_image_construct.start()





    def close_usb(self):
        self.thread_getMessage.terminate()
        self.thread_getMessage.join()

        self.thread_usbdecode.terminate()
        self.thread_usbdecode.join()

        self.thread_image_construct.terminate()
        self.thread_image_construct.join()


    def save(self):
        print('zhibin')


    def key_monitoring(self):
        keyboard.add_hotkey('c', self.save())  # 初始化验证
        keyboard.wait()









if __name__ == '__main__':
    usb = USB_DataDecode()

