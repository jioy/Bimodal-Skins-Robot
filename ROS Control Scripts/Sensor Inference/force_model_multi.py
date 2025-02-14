#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:38:11 2024

@author: Zhibin refurence Xinxin
"""
import rospy
from geometry_msgs.msg import WrenchStamped
import time
import numpy as np
import sys
import serial
import time
import struct
import os
import multiprocessing
from multiprocessing import Queue
import torch
import torch.nn as nn
import lib.Models as Models
import lib.skin_sensor as skin_sensor
import csv


##path set
print(sys.argv[0])
os.chdir(os.path.split(sys.argv[0])[0])
print(os.getcwd())


def force_pub(init_f):
    pub = rospy.Publisher('SRI_force_topic', WrenchStamped,queue_size = 10)
    rospy.init_node('SRI_node',anonymous=True)
    rate = rospy.Rate(50)
    
    while not rospy.is_shutdown():
        data = ser.read(1000)
        if len(data) > 30:
            if 0xAA == data[0] and 0x55 == data[1]:
                fx = struct.unpack('f', data[6:10])[0] - init_f[0]
                fy = struct.unpack('f', data[10:14])[0] - init_f[1]
                fz = struct.unpack('f', data[14:18])[0] - init_f[2]
                mx = struct.unpack('f', data[18:22])[0] - init_f[3]
                my = struct.unpack('f', data[22:26])[0] - init_f[4]
                mz = struct.unpack('f', data[26:30])[0] - init_f[5]
                #fff[0:6] = fx, fy, fz, mx, my, mz
                
        fs = WrenchStamped()
        fs.wrench.force.x = float(fx)
        fs.wrench.force.y = float(fy)
        fs.wrench.force.z = -float(fz)
        fs.wrench.torque.x = float(mx)
        fs.wrench.torque.y = float(my)
        fs.wrench.torque.z = float(mz)
        fs.header.stamp = rospy.Time.now()
        pub.publish(fs)
        
        sys.stdout.write('fx: %4f, fy: %4f, fz: %4f, mx: %4f, my: %4f, mz: %4f'%(fx, fy, -fz, mx, my, mz))
        sys.stdout.write('\r')
        sys.stdout.flush()
        rate.sleep()


#six force sensor
class SIXF_sensor:
    def __init__(self):
        super().__init__()
        self.ser = serial.Serial(port='/dev/ttyUSB2',
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS, timeout=1e-2)
        time.sleep(1e-2)  # sleep 100 ms
        
    
    def clear_Queue(self,q):
        res = []
        while q.qsize()>0:
            res.append(q.get())
    

    def init_sixforce(self):
        # 设置采样频率 100hz
        set_update_rate = "AT+SMPF=100\r\n".encode('utf-8')
        self.ser.write(bytearray(set_update_rate))
        # recvData3 = bytearray(ser.readall())
        # print(recvData3)
        # 上传数据格式
        set_recieve_format = "AT+SGDM=(A01,A02,A03,A04,A05,A06);E;1;(WMA:1)\r\n".encode('utf-8')
        self.ser.write(bytearray(set_recieve_format))
        get_data_once = "AT+GSD\r\n".encode('utf-8')
        self.ser.write(bytearray(get_data_once))
        self.init_f = np.zeros(6)
        fx_init = 0
        fy_init = 0
        fz_init = 0
        mx_init = 0
        my_init = 0
        mz_init = 0
        j = 0
        for i in range(200):
            init_data = self.ser.read(1000)
            if len(init_data) > 30:
                if 0xAA == init_data[0] and 0x55 == init_data[1]:
                    fx = struct.unpack('f', init_data[6:10])[0]
                    fy = struct.unpack('f', init_data[10:14])[0]
                    fz = struct.unpack('f', init_data[14:18])[0]
                    mx = struct.unpack('f', init_data[18:22])[0]
                    my = struct.unpack('f', init_data[22:26])[0]
                    mz = struct.unpack('f', init_data[26:30])[0]
                    # if fx and fy and fz and mx and my and mz:
                    fx_init = struct.unpack('f', init_data[6:10])[0] + fx_init
                    fy_init = struct.unpack('f', init_data[10:14])[0] + fy_init
                    fz_init = struct.unpack('f', init_data[14:18])[0] + fz_init
                    mx_init = struct.unpack('f', init_data[18:22])[0] + mx_init
                    my_init = struct.unpack('f', init_data[22:26])[0] + my_init
                    mz_init = struct.unpack('f', init_data[26:30])[0] + mz_init
                    j = j + 1
        fx_init = fx_init / j
        fy_init = fy_init / j
        fz_init = fz_init / j
        mx_init = mx_init / j
        my_init = my_init / j
        mz_init = mz_init / j
        self.init_f[0:6] = fx_init, fy_init, fz_init, mx_init, my_init, mz_init


    def read_sixforce(self,fix_queue):
        self.init_sixforce()
        forcedata = np.zeros(6)
        while True:
            data = self.ser.read(1000)
            if len(data) > 30:
                if 0xAA == data[0] and 0x55 == data[1]:
                    fx = struct.unpack('f', data[6:10])[0]
                    fy = struct.unpack('f', data[10:14])[0]
                    fz = struct.unpack('f', data[14:18])[0]
                    mx = struct.unpack('f', data[18:22])[0]
                    my = struct.unpack('f', data[22:26])[0]
                    mz = struct.unpack('f', data[26:30])[0]
                    #fff[0:6] = fx, fy, fz, mx, my, mz
                    forcedata[0:6] = fx, fy, fz, mx, my, mz
                    forcedata = forcedata - self.init_f
                    self.clear_Queue(fix_queue)
                    fix_queue.put(forcedata)
                    #print(forcedata)



class USB_sensor:
    def __init__(self):
        super().__init__()
       
    
    def clear_Queue(self,q):
        res = []
        while q.qsize()>0:
            res.append(q.get())
    

    def readdata(self,sensor_queue):
        getcount = 0
        start_time = int(time.time())
        self.skin_sensor = skin_sensor.skin_sensor(start_time) #电子皮肤

        initlist = []
        for i in range(20):
            initdata1 = np.array(self.skin_sensor.getdata_ser(self.skin_sensor.ser_skin1,0))
            initdata2 = np.array(self.skin_sensor.getdata_ser(self.skin_sensor.ser_skin2,1))
            initdata3 = np.array(self.skin_sensor.getdata_ser(self.skin_sensor.ser_skin3,2))
            init_alldata = np.concatenate((initdata2,initdata1,initdata3),axis=0)
            initlist.append(init_alldata)
        init_alldata = np.mean(np.array(initlist),axis = 0)

        
        while True:

            getdata1 = np.array(self.skin_sensor.getdata_ser(self.skin_sensor.ser_skin1,0))
            getdata2 = np.array(self.skin_sensor.getdata_ser(self.skin_sensor.ser_skin2,1))
            getdata3 = np.array(self.skin_sensor.getdata_ser(self.skin_sensor.ser_skin3,2))
            one_alldata = np.concatenate((getdata2,getdata1,getdata3),axis=0)

            deta_data = (one_alldata - init_alldata)*0.35




            getcount = getcount + 1
            sensor_queue.put(deta_data)

            # value = getdata1 + getdata2 + getdata3

            if(getcount>=40):
                skin_ticks = time.time()
                #print(str(skin_ticks))
                getcount = 0

    def save_data_init(self,filenname):
      headers = ['times','Fx_skin','Fz_skin','T_skin',
                'Fx_six','Fz_six','T_six']
      with open(filenname,'w') as form:
         writer = csv.writer(form)
         writer.writerow(headers)
         
    def save_data(self,data,filenname):
      with open(filenname,'a') as form:
         writer = csv.writer(form)
         writer.writerow(data)            

    def model_cal(self,sensor_queue,fix_queue):
        pub = rospy.Publisher('SRI_force_topic', WrenchStamped,queue_size = 10)
        rospy.init_node('SRI_node',anonymous=True)
        rate = rospy.Rate(50)

        savetime = int(time.time())
        #self.csvpath = './zhibin_lib/datasave/'+str(savetime)+'.csv'
        #self.save_data_init(self.csvpath)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ###Create the generator
        net = Models.TransformerEncoderModel(input_dim=768, num_heads=4, hidden_dim=256, output_dim=3, num_layers=1,
                                         dropout_rate=0.1).to(device)
        net = nn.DataParallel(net, list(range(1))) 
        save_path = './lib/model_K1.pth'
        net.load_state_dict(torch.load(save_path))  #save_path
        print(net)
        getcount = 0
        getsix = np.zeros(6)
        sensor_filter = torch.zeros((20,1,768),dtype=torch.float)
        sensor_filter = sensor_filter.to(device,dtype=torch.float)
        fout_filter = torch.zeros((5,3),dtype=torch.float)
        fout_filter = fout_filter.to(device,dtype=torch.float)
        while not rospy.is_shutdown():
            getsensor = sensor_queue.get(True)
            #print(getsensor.shape)
            sensor = torch.tensor(getsensor)
            sensor = sensor.view(1,1,768)
            sensor = sensor.to(device,dtype=torch.float)
            # sensor_filter = torch.roll(sensor_filter, 1, dims = 0)
            # sensor_filter[0,:,:] = sensor[0,:,:]
            # sensor = torch.mean(sensor_filter, dim = 0)
            # sensor = sensor.view(1,1,768)

            #out and filter
            outdata = net(sensor)
            #fout_filter = torch.roll(fout_filter, 1, dims = 0)
            #fout_filter[0,:] = outdata[0,:]
            #outdata = torch.mean(fout_filter, dim = 0)
            outdata = outdata.view(1,3)


            
            outdata = outdata.cpu().detach().numpy()
            

            getsix = [0,0,0,0,0,0,0,0,0,0,0] #fix_queue.get(True)
            self.clear_Queue(sensor_queue)
            FS_x = outdata[0][0]
            FS_z = outdata[0][1]
            FS_T = -outdata[0][2]
            print(FS_x,FS_T)
            
            savearray = [time.time(), FS_x, FS_z, FS_T,
                        getsix[1],getsix[2],getsix[5]]
            #self.save_data(savearray,self.csvpath)



            fs = WrenchStamped()
            fs.wrench.force.x = float(0)
            fs.wrench.force.y = -float(FS_x)
            fs.wrench.force.z = -float(FS_z)
            fs.wrench.torque.x = float(0)
            fs.wrench.torque.y = float(0)
            fs.wrench.torque.z = float(FS_T)
            fs.header.stamp = rospy.Time.now()
            pub.publish(fs)


            # getcount = getcount + 1
            # if(getcount>=100):
            #     skin_ticks = time.time()
            #     print(outdata,skin_ticks)
            #     getcount = 0



if __name__ == '__main__':
    
    sensor_queue = Queue()
    fix_queue = Queue()

    usb_sensor = USB_sensor()
    thread_usbsensor = multiprocessing.Process(target = usb_sensor.readdata, args = (sensor_queue,))
    thread_modelcal = multiprocessing.Process(target = usb_sensor.model_cal, args = (sensor_queue,fix_queue,))

    sixf = SIXF_sensor()
    thread_sixf = multiprocessing.Process(target = sixf.read_sixforce,args = (fix_queue,))


    thread_usbsensor.start()
    thread_modelcal.start()
    #thread_sixf.start()

    