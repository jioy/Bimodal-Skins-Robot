#!/usr/bin/env python
# coding=utf-8

#@author: Zhibin refurence Xinxin



import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import csv
import sys, select, termios, tty
import time, os


##path set
print(sys.argv[0])
os.chdir(os.path.split(sys.argv[0])[0])
print(os.getcwd())




msg = """
"""
Omni = 0 #全向移动模式
speed = 0.2 #默认移动速度 m/s
turn  = 0.5   #默认转向速度 rad/s
#以字符串格式返回当前速度
def vels(speed,turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

class force_sensor_velocity_controller:
    def __init__(self):
        self.Omni = 0
        
        rospy.init_node('force_to_velocity_controller')
        rospy.Subscriber('/SRI_force_topic',WrenchStamped,self.force_callback)
        rospy.Subscriber('/odom',Odometry,self.odom_callback)
        self.vel_pub = rospy.Publisher('~cmd_vel', Twist, queue_size=5) #创建速度话题发布者，'~cmd_vel'='节点名/cmd_vel'
        
        self.force_data = None
        self.odom_data = None
        self.Fy = 0;
        self.Fz = 0;
        self.Tz = 0;
        
        
    def force_callback(self, force_data):
        #print(force_data.wrench.force.y, force_data.wrench.force.z, force_data.wrench.torque.z)
        self.force_data = force_data.wrench
    
    def odom_callback(self, odom_data):
        #print(odom_data.twist.twist)
        self.odom_data = odom_data.twist.twist
   
    def cal_speed(self):
        if self.force_data is None or self.odom_data is None:
            return None
        
        current_speed = self.odom_data.linear.x
        current_turn = self.odom_data.angular.z
        
        self.Fy = self.force_data.force.y
        self.Fz = self.force_data.force.z
        self.Tz = self.force_data.torque.z
        #如果没有力交互，速度赋0
        if abs(self.force_data.force.x)+abs(self.force_data.force.y)+abs(self.force_data.force.z) < 8 and abs(self.force_data.torque.z) < 1.2:
            target_speed = 0
            target_turn = 0
        else:
            #target_speed = -self.force_data.force.y*0.02 + current_speed
	    target_speed = (-self.force_data.force.y*0.008-0.2*current_speed) + current_speed
            target_turn = (self.force_data.torque.z*0.05-0.2*current_turn) + current_turn
        
        #平滑控制，计算前进后退实际控制速度
        control_speed = np.clip(target_speed, -0.15, 0.8)
        control_speed = np.clip(control_speed, current_speed - 0.5, current_speed + 0.3)

        #平滑控制，计算转向实际控制速度
        control_turn = np.clip(target_turn, - 0.3, 0.8)
        control_turn = np.clip(control_turn, current_turn - 0.5, current_turn + 0.5)


        #根据是否全向移动模式，给速度话题变量赋值
        vel_cmd = Twist()
        if self.Omni==0:
            vel_cmd.linear.x  = control_speed; vel_cmd.linear.y = 0;  vel_cmd.linear.z = 0
            vel_cmd.angular.x = 0;             vel_cmd.angular.y = 0; vel_cmd.angular.z = control_turn
        else:
            vel_cmd.linear.x  = control_speed; vel_cmd.linear.y = 0; vel_cmd.linear.z = 0
            vel_cmd.angular.x = 0;             vel_cmd.angular.y = 0; vel_cmd.angular.z = 0
        
        return vel_cmd

    def save_data_init(self,filenname):
      headers = ['times','linear_x','T','Fx','Fz','Tz']
      with open(filenname,'w') as form:
         writer = csv.writer(form)
         writer.writerow(headers)
         
    def save_data(self,data,filenname):
      with open(filenname,'a') as form:
         writer = csv.writer(form)
         writer.writerow(data) 
        
    def run(self):
        rate = rospy.Rate(100)
        savetime = int(time.time())
        self.csvpath = './'+str(savetime)+'_speed.csv'
        self.save_data_init(self.csvpath)
        while not rospy.is_shutdown():
            vel_cmd = self.cal_speed()
            if vel_cmd:
                self.vel_pub.publish(vel_cmd)
                print(vel_cmd)
                savedata_list = [time.time(), vel_cmd.linear.x, vel_cmd.angular.z, -self.Fy, self.Fz, self.Tz]
                self.save_data(savedata_list, self.csvpath)
                rate.sleep()
            else:
                vel_cmd = Twist()
                vel_cmd.linear.x = 0;  vel_cmd.linear.y = 0;  vel_cmd.linear.z = 0
                vel_cmd.angular.x = 0; vel_cmd.angular.y = 0; vel_cmd.angular.z = 0
                self.vel_pub.publish(vel_cmd)
#主函数
if __name__=="__main__":
    controller = force_sensor_velocity_controller()
    controller.run()


