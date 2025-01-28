#ifndef __SYSSENSE_H
#define __SYSSENSE_H


typedef unsigned char u_int8_t;
typedef short int16_t;
typedef unsigned short u_int16_t;
typedef int int32_t;
typedef unsigned int u_int32_t;




/* 五个点的型号是：SYC2310-I   目前有3个设备，ID为： 1010A583  1010A584  1010A585
八个点的型号是：SYC2310-IIlite   目前有1个设备，ID为： 1011A691
256个点的型号是：SYC2310-II   目前有1个设备，ID为： 1011B361

 */	
#define MODEL_SYC2310I "SY2310-I"	/* 2310I 型号 5点*/	
#define MODEL_SYC2310IIlite "SYC2310-IIlite"	/* 2310I 型号 8点*/
#define MODEL_SYC2310II "SY2310-II"	/* 2310II 型号 256点*/	

#define Syc_multicast_IP "224.0.0.198" /* 同一路由器下，组播地址 */
#define Syc_multicast_Port_Dev 8061 /* 同一路由器下，组播地址 设备端口 */
#define Syc_multicast_Port 8091 /* 同一路由器下，组播地址 数据终端端口 */
#define Syc_Terminal_Eth_If "ens33" /* 默认eth0 */

#define Syc_Data_Cap_File "./SycDataCap.log" /* 数据文件 */
#define Syc_Data_Cap_Max_Num 0x00FF /* 模拟数据记录组数 */




#ifndef MODEL_SYC2310IIlite
#define MODEL_SYC2310IIlite
#endif

/* 传感器的个数5 */ 
#ifdef MODEL_SYC2310IIlite
#define Syc_Sensor_Num	8	/* 传感器的个数 */ 
#define Syc_Sensor_Data_Len 3		/* 整数部分字节数=小数部分字节数=Syc_Sensor_Data_Len，每个传感器的数据字节数2*Syc_Sensor_Data_Len */ 
#define Syc_Dev_Model "SYC2310-I" /* 正式量产应该写在flash中 */
#define Syc_Dev_Sn "1011A691" /* 正式量产应该写在flash中 也做DeviceID*/
#define Syc_Dev_Ver "HW1.0001;SW1.0002;231203" /* 正式量产应该写在flash中 */



#else
#define Syc_Sensor_Num	256	/* 传感器的个数 */ 
#define Syc_Sensor_Data_Len 3		/* 整数部分字节数=小数部分字节数=Syc_Sensor_Data_Len，每个传感器的数据字节数2*Syc_Sensor_Data_Len */ 
#define Syc_Dev_Model "SYC2310-II" /* 正式量产应该写在flash中 */
#define Syc_Dev_Sn "7A19555A" /* 正式量产应该写在flash中 */
#define Syc_Dev_Ver "HW1.0002;SW1.0001;231203" /* 正式量产应该写在flash中 */

#endif /* MODEL_SYC2310I */















#define Syc_Msg_Max_Len	Syc_Sensor_Num*2*Syc_Sensor_Data_Len		/* SY2310I Msg body buffer size */
#define Syc_Cmd_Len  12 /* 控制命令，无数据段，长度为：12 */
#define Syc_Cmd_Re_Len_long 42 /* 控制命令回复（有数据段） 用不上的记得初始化为NULL，最大长度定义为：42，字符串最多29个uint8，加一个NULL结尾*/
#define Syc_Cmd_Re_Len 12 /* 控制命令回复（无数据段）长度为：12 */


unsigned int hextoi(u_int8_t *hexstring);
int Syc_InttoMSGbody(unsigned char *m_Hex,int m_Int);
//状态机
int State_CMD(); //状态字解析
int State_Machine();





#define Syc_Hi 0x0001
#define Syc_Hi_Re 0x0002

#define Syc_Init 0x0003
#define Syc_Init_Re 0x0004

#define Syc_Sn 0x0005
#define Syc_Sn_Re 0x0006

#define Syc_Ver 0x0007
#define Syc_Ver_Re 0x0008

#define Syc_Reset 0x0009
#define Syc_Reset_Re 0x000A

#define Syc_Batt 0x000B
#define Syc_Batt_Re 0x000C

#define Syc_Data_Start 0x0011
#define Syc_Data_Start_Re 0x0012

#define Syc_Data_Stop 0x0013
#define Syc_Data_Stop_Re 0x0014

#define Syc_Data_Type 0x0015
#define Syc_Data_Type_Re 0x0016

#define Syc_Magic_Num 0xFFFF0609



/*1、 包头 */	
typedef struct __attribute__((packed)){ //包头长20个字节
                        u_int32_t magicNUM;
                        u_int16_t sequentialNUM;
                        u_int16_t length;
                        u_int32_t deviceID;
                        u_int32_t destinationID;
                        u_int16_t CMD;
                        u_int16_t STS;
} SycMSGHead;




/*2、数据 */	
typedef struct __attribute__((packed)){ //每个传感器的数据结构，例如：5个传感器的型号，数据宽度为3字节（整数小数位各位3个字节），包体就是30个字节
                        u_int8_t integer_Data[Syc_Sensor_Data_Len];
						u_int8_t decimal_Data[Syc_Sensor_Data_Len];
} SycMSGSensorData;




/*3、组合 */	
typedef struct __attribute__((packed)){ //每个命令的数据结构，例如：5个传感器的型号，数据位宽为3字节（整数小数位各位3个字节），包头+包体50个字节
    SycMSGHead MSGHead;
    SycMSGSensorData MSGbody[Syc_Sensor_Num];
} SycMSG;



#endif