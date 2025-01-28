#ifndef __SYSSENSE_H
#define __SYSSENSE_H


typedef unsigned char u_int8_t;
typedef short int16_t;
typedef unsigned short u_int16_t;
typedef int int32_t;
typedef unsigned int u_int32_t;




/* �������ͺ��ǣ�SYC2310-I   Ŀǰ��3���豸��IDΪ�� 1010A583  1010A584  1010A585
�˸�����ͺ��ǣ�SYC2310-IIlite   Ŀǰ��1���豸��IDΪ�� 1011A691
256������ͺ��ǣ�SYC2310-II   Ŀǰ��1���豸��IDΪ�� 1011B361

 */	
#define MODEL_SYC2310I "SY2310-I"	/* 2310I �ͺ� 5��*/	
#define MODEL_SYC2310IIlite "SYC2310-IIlite"	/* 2310I �ͺ� 8��*/
#define MODEL_SYC2310II "SY2310-II"	/* 2310II �ͺ� 256��*/	

#define Syc_multicast_IP "224.0.0.198" /* ͬһ·�����£��鲥��ַ */
#define Syc_multicast_Port_Dev 8061 /* ͬһ·�����£��鲥��ַ �豸�˿� */
#define Syc_multicast_Port 8091 /* ͬһ·�����£��鲥��ַ �����ն˶˿� */
#define Syc_Terminal_Eth_If "ens33" /* Ĭ��eth0 */

#define Syc_Data_Cap_File "./SycDataCap.log" /* �����ļ� */
#define Syc_Data_Cap_Max_Num 0x00FF /* ģ�����ݼ�¼���� */




#ifndef MODEL_SYC2310IIlite
#define MODEL_SYC2310IIlite
#endif

/* �������ĸ���5 */ 
#ifdef MODEL_SYC2310IIlite
#define Syc_Sensor_Num	8	/* �������ĸ��� */ 
#define Syc_Sensor_Data_Len 3		/* ���������ֽ���=С�������ֽ���=Syc_Sensor_Data_Len��ÿ���������������ֽ���2*Syc_Sensor_Data_Len */ 
#define Syc_Dev_Model "SYC2310-I" /* ��ʽ����Ӧ��д��flash�� */
#define Syc_Dev_Sn "1011A691" /* ��ʽ����Ӧ��д��flash�� Ҳ��DeviceID*/
#define Syc_Dev_Ver "HW1.0001;SW1.0002;231203" /* ��ʽ����Ӧ��д��flash�� */



#else
#define Syc_Sensor_Num	256	/* �������ĸ��� */ 
#define Syc_Sensor_Data_Len 3		/* ���������ֽ���=С�������ֽ���=Syc_Sensor_Data_Len��ÿ���������������ֽ���2*Syc_Sensor_Data_Len */ 
#define Syc_Dev_Model "SYC2310-II" /* ��ʽ����Ӧ��д��flash�� */
#define Syc_Dev_Sn "7A19555A" /* ��ʽ����Ӧ��д��flash�� */
#define Syc_Dev_Ver "HW1.0002;SW1.0001;231203" /* ��ʽ����Ӧ��д��flash�� */

#endif /* MODEL_SYC2310I */















#define Syc_Msg_Max_Len	Syc_Sensor_Num*2*Syc_Sensor_Data_Len		/* SY2310I Msg body buffer size */
#define Syc_Cmd_Len  12 /* ������������ݶΣ�����Ϊ��12 */
#define Syc_Cmd_Re_Len_long 42 /* ��������ظ��������ݶΣ� �ò��ϵļǵó�ʼ��ΪNULL����󳤶ȶ���Ϊ��42���ַ������29��uint8����һ��NULL��β*/
#define Syc_Cmd_Re_Len 12 /* ��������ظ��������ݶΣ�����Ϊ��12 */


unsigned int hextoi(u_int8_t *hexstring);
int Syc_InttoMSGbody(unsigned char *m_Hex,int m_Int);
//״̬��
int State_CMD(); //״̬�ֽ���
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



/*1�� ��ͷ */	
typedef struct __attribute__((packed)){ //��ͷ��20���ֽ�
                        u_int32_t magicNUM;
                        u_int16_t sequentialNUM;
                        u_int16_t length;
                        u_int32_t deviceID;
                        u_int32_t destinationID;
                        u_int16_t CMD;
                        u_int16_t STS;
} SycMSGHead;




/*2������ */	
typedef struct __attribute__((packed)){ //ÿ�������������ݽṹ�����磺5�����������ͺţ����ݿ��Ϊ3�ֽڣ�����С��λ��λ3���ֽڣ����������30���ֽ�
                        u_int8_t integer_Data[Syc_Sensor_Data_Len];
						u_int8_t decimal_Data[Syc_Sensor_Data_Len];
} SycMSGSensorData;




/*3����� */	
typedef struct __attribute__((packed)){ //ÿ����������ݽṹ�����磺5�����������ͺţ�����λ��Ϊ3�ֽڣ�����С��λ��λ3���ֽڣ�����ͷ+����50���ֽ�
    SycMSGHead MSGHead;
    SycMSGSensorData MSGbody[Syc_Sensor_Num];
} SycMSG;



#endif