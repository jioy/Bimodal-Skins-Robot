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
from vispy.io import imread
from vispy.visuals.filters import TextureFilter
from PIL import Image



class skin_sensor():


    def __init__(self, start_time):

        # 数据存储路径
        self.savepath = '/home/wheeltec/wheeltec_robot/src/wheeltec_robot_rc/scripts/Zhibin/exp_data/Label_twist/' + str(
            start_time) + '.csv'
        # 初始化数据保存

        # 参数：
        self.erro = 0
        self.datalen = (256 + 2) * 2  #
        self.datalen_Tem = (32 + 2) * 2  #
        self.time_start = time.time()
        self.get_count = 0

        self.received_erro = [0] * 256
        self.Temlist = [[0] * 25, [0] * 25, [0] * 25]




    def timevrify_ser(self, ser):
        count_num = ser.inWaiting()

        if count_num > self.datalen:

            count = ser.read(self.datalen)
            if count[0:2] == b'\xaa\xaa' and count[-2:] == b'\xbb\xbb':
                self.get_count += 1
                if (self.get_count >= 40):
                    self.get_count = 0
                    time_end = time.time()  #
                    time_sum = time_end - self.time_start  #
                    print('datatime:', time_sum)
                    self.time_start = time.time()  #

            else:
                count_num = ser.inWaiting()  #
                if count_num > 0:
                    count = ser.read(count_num)
                self.erro += 1
                print('erro---------------------------------------------------------' + str(self.erro))

    def getdata_ser(self, ser, IONUM):
        while True:
            count_num = ser.inWaiting()
            if count_num > self.datalen:
                count = ser.read(self.datalen)
                if count[0:2] == b'\xaa\xaa' and count[-2:] == b'\xbb\xbb':
                    received = []
                    for i in range(0, 256):
                        data = count[2 + i * 2: 4 + i * 2]
                        data_analysis = struct.unpack('<h', data)  # 元组  2字节
                        received.append(data_analysis[0])  # int.from_bytes(data, byteorder='little', signed=False)

                    return received

                else:
                    count_num = ser.inWaiting()  #
                    if count_num > 0:
                        count = ser.read(count_num)
                    self.erro += 1
                    print('erro---------------------------------------------------------' + str(self.erro) + 'P' + str(
                        IONUM))
                    continue
            else:
                continue  # time.sleep(0.001)

    def getdata_Tem(self, ser, IONUM):
        while True:
            count_num = ser.inWaiting()
            if count_num > self.datalen_Tem:
                count = ser.read(self.datalen_Tem)
                if count[0:2] == b'\xaa\xaa' and count[-2:] == b'\xbb\xbb':
                    received = []
                    for i in range(0, 25):
                        data = count[2 + i * 2: 4 + i * 2]
                        data_analysis = struct.unpack('<h', data)  # 元组  2字节
                        received.append(data_analysis[0])  # int.from_bytes(data, byteorder='little', signed=False)
                        self.Temlist[IONUM] = received
                    return received

                else:
                    count_num = ser.inWaiting()  #
                    if count_num > 0:
                        count = ser.read(count_num)
                    self.erro += 1
                    print('erro---------------------------------------------------------' + str(self.erro) + 'T' + str(
                        IONUM))
                    continue
            else:
                return self.Temlist[IONUM]

    # 清空队列
    def clear_Queue(self, q):
        res = []
        while q.qsize() > 0:
            res.append(q.get(True))


    #连续解析返回数据
    def continue_decode(self,data_raed):
        # 串口
        self.ser_skin1 = serial.Serial('com8', 2000000)
        self.ser_skin2 = serial.Serial('com7', 2000000)
        self.ser_skin3 = serial.Serial('com9', 2000000)
        if self.ser_skin1.isOpen() and self.ser_skin2.isOpen() and self.ser_skin3.isOpen():
            print("serial successs")
            print(self.ser_skin1.name)
            print(self.ser_skin2.name)
            print(self.ser_skin3.name)

        else:
            while True:
                print("fiald")
        while True:

            getdata1 = self.getdata_ser(self.ser_skin1, 0)
            getdata2 = self.getdata_ser(self.ser_skin2, 1)
            getdata3 = self.getdata_ser(self.ser_skin3, 2)
            #self.clear_Queue(data_raed)
            data_raed.put([getdata1, getdata2, getdata3])
            print('read')






class GUI_Rebuild():     #USB 400Hz 刷新率

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



    #开机求取平均值
    def init_data(self,data_raed):

        init_data_buffer = np.zeros((3,256), dtype=int)
        list_data_buffer = []
        for i in range(20):
            getdata = data_raed.get(True)  # 接收数据
            getdata_array = np.array(getdata)
            init_data_buffer = init_data_buffer + getdata_array
            list_data_buffer.append(getdata_array)
        init_data_buffer = np.ceil(init_data_buffer/self.coefficient)


        # list_data_buffer = np.array(list_data_buffer)
        #
        # for i in range(256):
        #     init_data_buffer[0,i] = np.max(list_data_buffer[:,0,i])
        #     init_data_buffer[1,i] = np.max(list_data_buffer[:,1,i])
        #     init_data_buffer[2,i] = np.max(list_data_buffer[:,2,i])
        # #mux_data = np.max(list_data_buffer[i])

        return init_data_buffer

    # 清空队列
    def clear_Queue(self, q):
        res = []
        while q.qsize() > 0:
            res.append(q.get(True))




    def image_reconstructed(self,data_raed,image_construct):
        self.init_array = self.init_data(data_raed)

        self.png = np.flipud(imread('./UVPNG/skin.png'))
        # print(self.png.shape)
        self.texture = np.flipud(imread('./UVPNG/skin1024.png'))

        sensor = np.zeros((1024, 1024))
        ave_value = 0
        for i in range(16):
            for j in range(16):
                sensor[(j) * 33 + 33, (i) * 40 + 40] = 500

        for i in range(16):
            for j in range(16):
                sensor[(j+15) * 33 + 33, (i) * 40 + 40] = 500


        sensor = sensor[:, ::-1]  # 列反转
        sensor = pg.gaussianFilter(sensor, (10, 10))  # 高斯平滑
        sensor *= 255  # 变换为0-255的灰度值
        sensor = sensor.astype(np.uint8)

        # 热力图
        heat_img = cv2.applyColorMap(sensor, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
        heat_img = np.roll(heat_img,300, axis=1)

        # # 使用格式
        # image = Image.fromarray(heat_img)  # 将之前的矩阵转换为图片
        # image.show()  # 调用本地软件显示图片，win10是叫照片的工具

        self.flag = 0

        while True:

            sensor = np.zeros((1024, 1024))
            getdata_Z = data_raed.get(True)  # 接收数据
            self.clear_Queue(data_raed)
            for i in range(16):
                for j in range(16):
                    pic_data = getdata_Z[0][(15-i)*16 + (15-j)]/self.coefficient - self.init_array[0][(15-i)*16 + (15-j)]
                    pic_data2 = getdata_Z[1][i * 16 + (15 - j)]/self.coefficient - self.init_array[1][i * 16 + (15 - j)]
                    if (pic_data < 50) : pic_data = 0
                    if (pic_data2 < 50): pic_data2 = 0

                    sensor[(j) * 33 + 33, (i) * 40 + 40] = pic_data  * 10
                    sensor[(j + 15) * 33 + 33, (i) * 40 + 40] = pic_data2  * 10

            # for i in range(16):
            #     for j in range(16):
            #         sensor[(j + 15) * 33 + 33, (i) * 40 + 40] = 500


            sensor = sensor[:, ::-1]  # 列反转
            sensor = pg.gaussianFilter(sensor, (30, 30))  # 高斯平滑
            sensor *= 255  # 变换为0-255的灰度值
            sensor = sensor.astype(np.uint8)

            # 热力图
            heat_img = cv2.applyColorMap(sensor, cv2.COLORMAP_CIVIDIS)  # 注意此处的三通道热力图是cv2专有的GBR排列
            heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
            heat_img = np.roll(heat_img, 300, axis=1)

            # 使用格式
            # image = Image.fromarray(heat_img)  # 将之前的矩阵转换为图片
            # image.show()  # 调用本地软件显示图片，win10是叫照片的工具


            # self_skindata = data_z.get(True)  # 接收数据
            # print(self_skindata[2])

            #time.sleep(0.1)
            #self.clear_Queue(image_construct)
            image_construct.put(heat_img)

            # if self.flag ==0:
            #     image_construct.put(heat_img)
            #     self.flag = 1
            # elif self.flag ==1:
            #     image_construct.put(heat_img)
            #     self.flag = 0
            # #self.flag = self.flag + 1
            #
            # pass












class USB_DataDecode():

    def __init__(self):


        #多进程
        self.data_raed = Queue() #数据读取更新

        self.image_construct = Queue()  #图像重建结果

        self.GUI_order = Queue()  # GUI控制数据解析


        #进程1 接收数据
        start_time = int(time.time())
        self.Getdata = skin_sensor(start_time)  # 电子皮肤
        self.thread_getData = multiprocessing.Process(target=self.Getdata.continue_decode,args=(self.data_raed,))

        # #进程2 图像重建
        self.Image_construct = GUI_Rebuild()
        self.thread_image_construct = multiprocessing.Process(target=self.Image_construct.image_reconstructed, args=(
        self.data_raed, self.image_construct))

        #self.thread_key_monitoring = multiprocessing.Process(target=self.key_monitoring, args=())


        #self.eat_process = multiprocessing.Process(target=self.eat, args=(3, "giao"))
        print("主进程ID:", os.getpid())
        self.thread_getData.start()
        self.thread_image_construct.start()
        # self.thread_image_construct.start()





    def close_usb(self):
        self.thread_getMessage.terminate()
        self.thread_getMessage.join()

        self.thread_usbdecode.terminate()
        self.thread_usbdecode.join()

        self.thread_image_construct.terminate()
        self.thread_image_construct.join()









if __name__ == '__main__':
    USB_DataDecode()


