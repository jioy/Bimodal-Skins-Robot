#include "SysSense.h"
#include "stdio.h" /*添加头文件 */
#include "main.h"

extern char hand_buff[50]; //定义接收字符串的缓冲区，可以自定义大小
extern UART_HandleTypeDef huart1;
//建立数据帧
extern SycMSG Send_SycMSG;
extern SycMSG Get_SycMSG;
extern uint16_t main_state;
/******************************************************************************/
/*
 *      convert a hex string to an integer. The end of the string or a non-hex
 *      character will indicate the end of the hex specification.
 */

unsigned int hextoi(u_int8_t *hexstring)
{
        register u_int8_t *h;
        register unsigned int   c, v;

        v = 0;
        h = hexstring;
        if (*h == '0' && (*(h+1) == 'x' || *(h+1) == 'X')) {
                h += 2;
        }
        while ((c = (unsigned int)*h++) != 0) {
                if (c >= '0' && c <= '9') {
                        c -= '0';
                } else if (c >= 'a' && c <= 'f') {
                        c = (c - 'a') + 10;
                } else if (c >=  'A' && c <= 'F') {
                        c = (c - 'A') + 10;
                } else {
                        break;
                }
                v = (v * 0x10) + c;
        }
        return v;
}



int Syc_InttoMSGbody(unsigned char *m_Hex,int m_Int)
{
     if(m_Int > 0xFFFFFF)
        {
                m_Int = 0xFFFFFF;
        }
         m_Hex[0] = (m_Int & 0xFF0000) >> 16;
			//printf("m_Hex[0]: %x\n", m_Hex[0]);
         m_Hex[1] = (m_Int & 0x00FF00) >> 8;
			//printf("m_Hex[1]: %x\n", m_Hex[1]);
         m_Hex[2] = (m_Int & 0x0000FF);
			//printf("m_Hex[2]: %x\n", m_Hex[2]);

}





//状态机响应新版状态字解析
int State_CMD()//传入接收缓存区
{
	char Send_txbuff[100] = {0}; //定义发送Buffer
	static uint16_t frame_length = 20;
	
//	sprintf(Send_txbuff, "Syc-hi0001\r\n");
//	HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
	//主函数运行状态清零
	main_state = 0x0000;
	
	
	Send_SycMSG.MSGHead.length = __REV16(20-8); //帧长度
	switch(__REV16(Get_SycMSG.MSGHead.CMD)){
	
		/* 命令字回复解析 */
		/* 1查询 */
		case 0x0001:
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//清空数据段
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
	    Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //状态回复成功
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0001;
			
		/* 3初始化 */	
		case 0x0003:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(Syc_Sensor_Num); //状态回复成传感器个数
			//HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, sizeof(Send_SycMSG));
			main_state = 0x0003;
			return 0x0003;
			
		/* 5序列号读取 */	
		case 0x0005:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //状态回复成功
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0005;
			
		/* 7版本号读取 */	
		case 0x0007:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//清空数据段
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //状态回复成功
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0007;
		
		/* 9重启 */	
		case 0x0009:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//清空数据段
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //状态回复成功
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0009;
			
			
		/* B电量读取 */	
		case 0x000B:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//清空数据段
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //状态回复成功
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x000B;
			
			
		/* 11开始采集 */	
		case 0x0011:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//清空数据段
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //状态回复成功
			//HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			
			main_state = 0x0011;
			return 0x0011;	
			
		/* 13停止采集 */	
		case 0x0013:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//清空数据段
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //状态回复成功
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			main_state = 0x0000;
			return 0x0013;
		
		/* 15数据长度字节 */	
		case 0x0015:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//清空数据段
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0x0003); //状态回复成功
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0015;
		
		default:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//清空数据段
			Send_SycMSG.MSGHead.CMD = __REV16(0xFFFF); //命令字回复
			Send_SycMSG.MSGHead.STS = __REV16(0xFFFF); //状态回复失败
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0404;
			
	}
}









//状态机响应
int State_Machine()//传入接收缓存区
{
	char Send_txbuff[200] = {0}; //定义发送Buffer

	
	//0001 设备查询
	if(strcmp("Syc_Hi",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-hi0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x1;
		}
	
	//0003 设备查询
	else if(strcmp("Syc_Init",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-init0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x3;
		}
		
	//0005 序列号查询
	else if(strcmp("Syc-sn",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-sn0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x5;
		}	

	//0007 版本号查询
	else if(strcmp("Syc-ver",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-ver0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x7;
		}	

	//0009 重启
	else if(strcmp("Syc-reset",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-reset0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x9;
		}	
		
	//000B 电量查询
	else if(strcmp("Syc-batt",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-batt0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0xB;
		}
		
		
	//0011 数据采集开始
	else if(strcmp("Syc-data-start",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-data-start\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x11;
		}		

	//0013 数据采集停止
	else if(strcmp("Syc-data-stop",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-data-stop\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x13;
		}		

	//0015 数据长度读取命令
	else if(strcmp("Syc-data-type",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-data-type\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x13;
		}

		else
		{
			sprintf(Send_txbuff, "Erro\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return -204;
		}		
}
