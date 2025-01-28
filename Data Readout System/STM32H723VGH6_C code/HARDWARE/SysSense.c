#include "SysSense.h"
#include "stdio.h" /*���ͷ�ļ� */
#include "main.h"

extern char hand_buff[50]; //��������ַ����Ļ������������Զ����С
extern UART_HandleTypeDef huart1;
//��������֡
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





//״̬����Ӧ�°�״̬�ֽ���
int State_CMD()//������ջ�����
{
	char Send_txbuff[100] = {0}; //���巢��Buffer
	static uint16_t frame_length = 20;
	
//	sprintf(Send_txbuff, "Syc-hi0001\r\n");
//	HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
	//����������״̬����
	main_state = 0x0000;
	
	
	Send_SycMSG.MSGHead.length = __REV16(20-8); //֡����
	switch(__REV16(Get_SycMSG.MSGHead.CMD)){
	
		/* �����ֻظ����� */
		/* 1��ѯ */
		case 0x0001:
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//������ݶ�
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
	    Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //״̬�ظ��ɹ�
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0001;
			
		/* 3��ʼ�� */	
		case 0x0003:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(Syc_Sensor_Num); //״̬�ظ��ɴ���������
			//HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, sizeof(Send_SycMSG));
			main_state = 0x0003;
			return 0x0003;
			
		/* 5���кŶ�ȡ */	
		case 0x0005:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //״̬�ظ��ɹ�
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0005;
			
		/* 7�汾�Ŷ�ȡ */	
		case 0x0007:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//������ݶ�
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //״̬�ظ��ɹ�
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0007;
		
		/* 9���� */	
		case 0x0009:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//������ݶ�
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //״̬�ظ��ɹ�
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0009;
			
			
		/* B������ȡ */	
		case 0x000B:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//������ݶ�
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //״̬�ظ��ɹ�
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x000B;
			
			
		/* 11��ʼ�ɼ� */	
		case 0x0011:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//������ݶ�
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //״̬�ظ��ɹ�
			//HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			
			main_state = 0x0011;
			return 0x0011;	
			
		/* 13ֹͣ�ɼ� */	
		case 0x0013:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//������ݶ�
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0x0001); //״̬�ظ��ɹ�
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			main_state = 0x0000;
			return 0x0013;
		
		/* 15���ݳ����ֽ� */	
		case 0x0015:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//������ݶ�
			Send_SycMSG.MSGHead.CMD = __REV16(__REV16(Get_SycMSG.MSGHead.CMD)+1); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0x0003); //״̬�ظ��ɹ�
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0015;
		
		default:
			Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //֡������
			memset(&Send_SycMSG.MSGbody, 0, sizeof(Send_SycMSG.MSGbody));//������ݶ�
			Send_SycMSG.MSGHead.CMD = __REV16(0xFFFF); //�����ֻظ�
			Send_SycMSG.MSGHead.STS = __REV16(0xFFFF); //״̬�ظ�ʧ��
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, 20);
			return 0x0404;
			
	}
}









//״̬����Ӧ
int State_Machine()//������ջ�����
{
	char Send_txbuff[200] = {0}; //���巢��Buffer

	
	//0001 �豸��ѯ
	if(strcmp("Syc_Hi",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-hi0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x1;
		}
	
	//0003 �豸��ѯ
	else if(strcmp("Syc_Init",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-init0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x3;
		}
		
	//0005 ���кŲ�ѯ
	else if(strcmp("Syc-sn",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-sn0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x5;
		}	

	//0007 �汾�Ų�ѯ
	else if(strcmp("Syc-ver",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-ver0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x7;
		}	

	//0009 ����
	else if(strcmp("Syc-reset",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-reset0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x9;
		}	
		
	//000B ������ѯ
	else if(strcmp("Syc-batt",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-batt0001\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0xB;
		}
		
		
	//0011 ���ݲɼ���ʼ
	else if(strcmp("Syc-data-start",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-data-start\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x11;
		}		

	//0013 ���ݲɼ�ֹͣ
	else if(strcmp("Syc-data-stop",hand_buff)==0)
		{
			sprintf(Send_txbuff, "Syc-data-stop\r\n");
			HAL_UART_Transmit_IT(&huart1, (uint8_t*)Send_txbuff, sizeof(Send_txbuff));
			return 0x13;
		}		

	//0015 ���ݳ��ȶ�ȡ����
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
