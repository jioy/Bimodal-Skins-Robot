#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Read Skin sensor data
==============
**Author**: `zhibin Li`
'''

import serial
import time
import struct
import csv
import pickle
import threading
import time

# print('start')

class skin_sensor():
    ser_skin1 = serial.Serial('/dev/ttyskinP0', 2000000)
    ser_skin2 = serial.Serial('/dev/ttyskinP1', 2000000)
    ser_skin3 = serial.Serial('/dev/ttyskinP2', 2000000)

      
    flag_data = 0

    def __init__(self,start_time):
        #super(skin_sensor, self).__init__()
        #数据存储路径
        self.savepath = '/home/wheeltec/wheeltec_robot/src/wheeltec_robot_rc/scripts/Zhibin/exp_data/Label_twist/' + str(start_time) + '.csv'
        #初始化数据保存
        #self.save_data_init(self.savepath)
        
        #参数：
        self.erro = 0
        self.datalen = (256 + 2) * 2  #
        self.datalen_Tem = (32 + 2) * 2  #
        self.time_start = time.time()
        self.get_count = 0

        self.serial_init()  # 连接初始化
        self.received_erro = [0]*256
        self.Temlist = [[0]*25,[0]*25,[0]*25]
        
       


    def serial_init(self):
        if skin_sensor.ser_skin1.isOpen() and skin_sensor.ser_skin2.isOpen() and skin_sensor.ser_skin3.isOpen():
            print("serial successs")
            print(skin_sensor.ser_skin1.name)
            print(skin_sensor.ser_skin2.name)
            print(skin_sensor.ser_skin3.name)
           
        else:
            while True:
                print("fiald")




    def timevrify_ser(self,ser):
        count_num = ser.inWaiting()

        if count_num > self.datalen:

            count = ser.read(self.datalen)
            if count[0:2] == b'\xaa\xaa' and count[-2:] == b'\xbb\xbb':
              self.get_count+=1
              if(self.get_count>=40):
                self.get_count = 0
                time_end = time.time()  #
                time_sum = time_end - self.time_start  #
                print('datatime:',time_sum)
                self.time_start = time.time()  #

            else:
              count_num = ser.inWaiting()  #
              if count_num > 0:
                  count = ser.read(count_num)
              self.erro += 1
              print('erro---------------------------------------------------------' + str(self.erro))


    def getdata_ser(self,ser,IONUM):
        while True:
            count_num = ser.inWaiting()
            if count_num > self.datalen:
                count = ser.read(self.datalen)
                if count[0:2] == b'\xaa\xaa' and count[-2:] == b'\xbb\xbb':
                    received = []
                    for i in range(0, 256):
                        data = count[2 + i*2 : 4 + i*2]
                        data_analysis = struct.unpack('<h', data)  # 元组  2字节
    
                        received.append(data_analysis[0])    #int.from_bytes(data, byteorder='little', signed=False)

                    return received

                else:
                    count_num = ser.inWaiting()  #
                    if count_num > 0:
                      count = ser.read(count_num)
                    self.erro += 1
                    print('erro---------------------------------------------------------' + str(self.erro) + 'P' +  str(IONUM))
                    continue
            else: 
              continue #time.sleep(0.001)
            
    def getdata_Tem(self,ser,IONUM):
        while True:
            count_num = ser.inWaiting()
            if count_num > self.datalen_Tem:
                count = ser.read(self.datalen_Tem)
                if count[0:2] == b'\xaa\xaa' and count[-2:] == b'\xbb\xbb':
                    received = []
                    for i in range(0, 25):
                        data = count[2 + i*2 : 4 + i*2]
                        data_analysis = struct.unpack('<h', data)  # 元组  2字节
                        received.append(data_analysis[0])    #int.from_bytes(data, byteorder='little', signed=False)
                        self.Temlist[IONUM] = received
                    return received

                else:
                    count_num = ser.inWaiting()  #
                    if count_num > 0:
                      count = ser.read(count_num)
                    self.erro += 1
                    print('erro---------------------------------------------------------' + str(self.erro) +'T'+ str(IONUM))
                    continue
            else: return self.Temlist[IONUM]                                 
             

    def save_data_init(self,filenname):
      time_head = ['times']
      data_head = ['sensor'+ str(i) for i in range(1,769)]
      headers = time_head + data_head
      with open(filenname,'w') as form:
         writer = csv.writer(form)
         writer.writerow(headers)
         
    def save_data(self,data):
      filenname = self.savepath
      with open(filenname,'a') as form:
         writer = csv.writer(form)
         writer.writerow(data)
         
         




def change_user():
    skin_ticks = time.time()
    delay_time = 25 - (int(skin_ticks*1000)-init_time)%25
    t = threading.Timer(0.001*delay_time + 0.001, change_user)
    t.start()
    savedata = [skin_ticks] + value + twist_save.getvalue + twist_save.IMUvalue
    with open(savepath_bat, 'ab') as f:
        pickle.dump(savedata, f)
    
    



if __name__ == '__main__':
    start_time = int(time.time())
    skin_sensor = skin_sensor(start_time) #电子皮肤
    time_start = time.time()
    getcount = 0
    
   
    getdata1 = skin_sensor.getdata_ser(skin_sensor.ser_skin1,0)
    getdata2 = skin_sensor.getdata_ser(skin_sensor.ser_skin2,1)
    getdata3 = skin_sensor.getdata_ser(skin_sensor.ser_skin3,2)


    getcount = getcount + 1
    value = getdata1 + getdata2 + getdata3

    # #每过n秒切换一次
    # init_time = int(time.time() * 1000)
    # t = threading.Timer(0.01, change_user)    
    # t.start()
    
    
    
    while True:
        getdata1 = skin_sensor.getdata_ser(skin_sensor.ser_skin1,0)
        getdata2 = skin_sensor.getdata_ser(skin_sensor.ser_skin2,1)
        getdata3 = skin_sensor.getdata_ser(skin_sensor.ser_skin3,2)
        getcount = getcount + 1
        value = getdata1 + getdata2 + getdata3

        if(getcount>=40):
            skin_ticks = time.time()
            print(str(skin_ticks))
            #时间测试
            time_end = time.time()  #
            time_sum = time_end - time_start  #
            # print('datatime:', time_sum)
            time_start = time.time()  #
            getcount = 0
