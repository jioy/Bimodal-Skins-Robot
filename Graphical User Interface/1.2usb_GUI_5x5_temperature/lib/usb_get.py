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
                data = self.com.read((32+2)*2)

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
        init_data_buffer = [0 for i in range(0,32)]
        data_buffer = {n: [] for n in range(32)}

        for i in range(20):

            udp_data = data_flag.get(True)  # 接收数据

            for i in range(0, 32):
                data_analysis = struct.unpack('<h', udp_data[2 + i * 2: 4 + i * 2])  # 元组  2字节
                data_decode = data_analysis[0] * self.coefficient

                data_buffer[i].append(data_decode)


        for i in range(32):
            init_data_buffer[i] = np.max(data_buffer[i])

        return init_data_buffer



    # 膝关节重建 图像获取
    def get_image(self):
        image_dir = r"./lib/AI/hand.png"
        self.joint = Image.open(image_dir)  # 打开图片
        self.joint = self.joint.convert("RGB") #变RGB,三通道
        self.joint = self.joint.resize((600, 600))
        self.joint = self.joint.rotate(270)  # 逆时针旋转90

        self.joint_h = self.joint.convert('L')
        self.joint = np.asarray(self.joint)  # 转换为矩阵
        self.joint_h = np.asarray(self.joint_h)  # 转换为矩阵

        self.coord_A = [[501,276],[468,356],[368,70],[364,154],[356,242],
                        [328,341], [232,341], [329,418], [232,417], [330,505],
                        [233,506], [267,52], [268,131], [270,219], [171,72],
                        [182,156], [198,247], [91,155], [115,238], [138,314]
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

        while True:

            get_z = data_z.get(True)[0] #接收数据
            self.clear_Queue(data_z)


            sensor = np.full((640,640),0) #np.zeros((600, 600))



            for i in range(5):
                for j in range(5):
                    if(get_z[i*5+j]) < 2:
                        get_z[i * 5 + j] = 0
                    sensor[(j)*108 + 108,(i)*108 + 108] = get_z[i*5+j]*0.3 + 20

            # for i in range(20):
            #     sensor[self.coord_A[i][0], self.coord_A[i][1]] = -get_z[i] + 1000




            sensor = sensor[:, ::-1]  # 列反转
            sensor = pg.gaussianFilter(sensor, (45, 45))  # 高斯平滑
            #print(sensor)
            sensor *= 255  # 变换为0-255的灰度值
            sensor = sensor.astype(np.uint8)
            sensor = np.rot90(sensor, 1)

            # 热力图

            heat_img = cv2.applyColorMap(sensor, cv2.COLORMAP_CIVIDIS)  # 注意此处的三通道热力图是cv2专有的GBR排列
            heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像


            #plotdata = sensor


            image_construct.put(heat_img)


    def usb_decode(self,data_flag,data_out,data_z,GUI_order):
        data_buffer = [0] * 32

        z = np.zeros((16, 16))
        Sensor = np.zeros((16, 16))

        send_flag = 0  #装载数据标志位
        strat_time = time.time()  #采样计时

        #初始化，求均值
        init_data = self.init_data(data_flag)



        while True:
            udp_data = data_flag.get(True) #接收数据

            for i in range(0, 32):
                data_analysis = struct.unpack('<h', udp_data[2 + i * 2: 4 + i * 2])  # 元组  2字节
                data_decode = data_analysis[0] * self.coefficient - init_data[i]

                #if (data_decode < 200):  data_decode = 0

                data_buffer[i] = data_decode

            # for i in range(16):
            #     for j in range(16):
            #         z[j , i] = data_buffer[i*16 + j]
            #         Sensor[j,i] = data_buffer[i*16 + j]


            data_z.put([data_buffer,0])

            #是否保存数据
            if (GUI_order.empty() == False):   #接收到采样标志后开始采样
                get_flag = GUI_order.get()
                if(get_flag == 'start'): #开始采样装入
                    self.clear_Queue(data_out)  #清空缓存
                    strat_time = time.time()
                    send_flag = 1
                if (get_flag == 'stop'):  # 开始采样装入
                    deta_time = time.time() - strat_time
                    print(deta_time)
                    send_flag = 0


            if(send_flag == 1):
                data_buffer = np.array(data_buffer) / self.coefficient
                data_buffer[0] = int(time.time())
                data_out.put(data_buffer)








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

