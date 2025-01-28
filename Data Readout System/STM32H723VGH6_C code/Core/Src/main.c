/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdio.h" /*添加头文件 */
#include <stdlib.h>
#include "arm_math.h"
#include "SysSense.h" /*数据帧头文件 */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */


#define AD9833_HANDLE hspi1

#define FFT_LENGTH_EXP 10
#define ADC_CHANNEL 8

#define AD7606_RESULT_1()	*(__IO uint16_t *)0x60000000
#define AD7606_RESULT_2()	*(__IO uint16_t *)0x64000000

#define max(a,b) (a>b?a:b)

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

const int16_t FFT_LENGTH = 1 << FFT_LENGTH_EXP;

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

SPI_HandleTypeDef hspi1;

TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart1;

NOR_HandleTypeDef hnor1;
NOR_HandleTypeDef hnor2;

/* USER CODE BEGIN PV */

int16_t sentdata[514];
int16_t sentdata_1x8[16];
int16_t sentdata_16x16[258];
int16_t testsentdata[128];
int16_t sentdata_decodeall[514];


int16_t sent_num_flag = 0;
static int AD1_NUM = 0;
static int AD2_NUM = 0;

//int16_t AD1_median[3];

float32_t AD1_buffer[8][FFT_LENGTH];
float32_t AD1_FFTINPUT[FFT_LENGTH];
float32_t AD2_buffer[8][FFT_LENGTH];
float32_t AD2_FFTINPUT[FFT_LENGTH];
float32_t window_func[FFT_LENGTH];


static float32_t testInput_f32[FFT_LENGTH];
static float32_t testOutput_f32[FFT_LENGTH];
static float32_t testOutputMag_f32[FFT_LENGTH];

uint8_t rxbuff[200] = {0}; //定义接收字符串的缓冲区，可以自定义大小
char hand_buff[50] = {0}; //定义接收字符串的缓冲区，可以自定义大小
char response_buff[200] = {0};
uint8_t uart_decodebuff[200] = {0};
uint8_t offset_index; //接收字符串缓冲区的下标及大小
uint8_t dot_index;
uint8_t uart1_rxdata[1];
uint16_t main_state;

/* VOFA软件接收 */
float VOFA_Float[4];       //定义的临时变量
uint8_t VOFA_Data[20];      //定义的传输Buffer




//建立数据帧
SycMSG Send_SycMSG;
SycMSG Get_SycMSG;
//SycMSG *Get_SycMSG_p = &Get_SycMSG;//指针p直接指向结构体


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_SPI1_Init(void);
static void MX_FMC_Init(void);
static void MX_TIM3_Init(void);
static void MX_NVIC_Init(void);
/* USER CODE BEGIN PFP */

//USB 自动枚举
void USB_Close(void); //USB 自动枚举

static void MX3090_Init(void); // wifi透传初始化

//译码器控制
void MUX_4_16(uint8_t channel);

void for_delay_us(uint32_t nus);

//AD9833 DDS
void AD9833_Write(SPI_HandleTypeDef* spi_handle, unsigned short TxData);
void AD9833_Config(SPI_HandleTypeDef* spi_handle, unsigned char reset, unsigned char sleepMode, unsigned char waveform);
void AD9833_SetFreq(SPI_HandleTypeDef* spi_handle, double freq);

//窗函数
void Init_window_func(float32_t* win_func, int length, int type);//窗函数初始化

//FFT
static int arm_rfft_f32_32T(uint8_t ad_label, uint8_t ad_channel, float32_t* ad_array);


////状态机
//int State_CMD(); //状态字解析
//int State_Machine();

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	char transStr[60];
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
	USB_Close();//关断USB，自动枚举
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_SPI1_Init();
  MX_FMC_Init();
  MX_TIM3_Init();
  MX_USB_DEVICE_Init();

  /* Initialize interrupts */
  MX_NVIC_Init();
  /* USER CODE BEGIN 2 */
	//1、使能ADC输出 200KHz
	HAL_TIM_PWM_Start(&htim3,TIM_CHANNEL_4);
	
	//2、ADC初始化 复位
	HAL_GPIO_WritePin(GPIOC, AD_RESET_Pin, GPIO_PIN_SET);
	HAL_Delay(10);
	HAL_GPIO_WritePin(GPIOC, AD_RESET_Pin, GPIO_PIN_RESET);
	HAL_Delay(10);
	
	//3、配置DDS——Configure DDS(DAC)

	HAL_GPIO_WritePin(MUX_CS_GPIO_Port, MUX_CS_Pin, GPIO_PIN_RESET); //使能MUX
	
	HAL_Delay(200);
	MUX_4_16(3);
	AD9833_SetFreq(&AD9833_HANDLE, 20000); //1MHz
  AD9833_Config(&AD9833_HANDLE, 0, 0, 0); // output sine
	MUX_4_16(2);
	AD9833_SetFreq(&AD9833_HANDLE, 21000); //1MHz
  AD9833_Config(&AD9833_HANDLE, 0, 0, 0); // output sine
	MUX_4_16(1);
	AD9833_SetFreq(&AD9833_HANDLE, 22000); //1MHz
  AD9833_Config(&AD9833_HANDLE, 0, 0, 0); // output sine
	MUX_4_16(0);
	AD9833_SetFreq(&AD9833_HANDLE, 23000); //1MHz
  AD9833_Config(&AD9833_HANDLE, 0, 0, 0); // output sine
	
	MUX_4_16(3);
	AD9833_SetFreq(&AD9833_HANDLE, 20000); //1MHz
  AD9833_Config(&AD9833_HANDLE, 0, 0, 0); // output sine
	
	for (uint8_t dds_num = 0;dds_num<12;dds_num++)
	{
		MUX_4_16(dds_num + 4);
		HAL_Delay(1);
		AD9833_SetFreq(&AD9833_HANDLE, 24000 + dds_num*1000); //1MHz
		AD9833_Config(&AD9833_HANDLE, 0, 0, 0); // output sine
		HAL_Delay(1);
	}
	HAL_GPIO_WritePin(MUX_CS_GPIO_Port, MUX_CS_Pin, GPIO_PIN_SET); //关闭MUX
	
//	//DDS4   暂时关闭
//	HAL_GPIO_WritePin(AD9833_CS4_GPIO_Port, AD9833_CS4_Pin, GPIO_PIN_RESET);
//	AD9833_SetFreq(&AD9833_HANDLE, 23000); //1MHz
//	AD9833_Config(&AD9833_HANDLE, 0, 0, 0); // output sine
//	HAL_GPIO_WritePin(AD9833_CS4_GPIO_Port, AD9833_CS4_Pin, GPIO_PIN_SET);

	
	
	
	//5、初始化窗函数
	Init_window_func(window_func, FFT_LENGTH, 3);
	
	//6、串口接收中断
	HAL_UART_Receive_IT(&huart1, (uint8_t*)uart1_rxdata, sizeof(uart1_rxdata));
	
	//7、建立数据帧
	memset(&Send_SycMSG, 0, sizeof(Send_SycMSG));
	memset(&Get_SycMSG, 0, sizeof(Get_SycMSG));
	//发送数据帧
	Send_SycMSG.MSGHead.magicNUM = __REV(Syc_Magic_Num); //1帧头
	Send_SycMSG.MSGHead.sequentialNUM = __REV16(0);//2、帧序列号
	Send_SycMSG.MSGHead.deviceID = __REV(hextoi(Syc_Dev_Sn));//4设备ID
	//Get_SycMSG.MSGHead.deviceID = Syc_Dev_Sn;
  //HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, sizeof(Send_SycMSG));
	
	int i = 0;
	
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	sentdata_decodeall[0] = 0xAAAA;
	sentdata_decodeall[513] = 0xBBBB;
	sentdata_16x16[0] = 0xAAAA;
	sentdata_16x16[257] = 0xBBBB;
	
	sentdata_1x8[0] = 0xAAAA;
	sentdata_1x8[15] = 0xBBBB;
	
	VOFA_Data[16] = 0x00;
	VOFA_Data[17] = 0x00;
	VOFA_Data[18] = 0x80;
	VOFA_Data[19] = 0x7f;
	
	//开启状态：
	main_state = 0;
	
	
  
  //HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, sizeof(Send_SycMSG));
  //	Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM) + 1);//帧序列递增
  //	HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, sizeof(Send_SycMSG));
  //	HAL_Delay(1000);
	
	
	uint8_t testbuf[128] = {0};
	for (i=0;i<128;i++)
	{
		testbuf[i] = i;
	}
	
	main_state = 0x0404;
	
	
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		
		
		 
		switch(main_state){
		
		  /* 按键启动配网 */
			case 1818:
				main_state = 0;
				__HAL_UART_DISABLE_IT(&huart1,UART_IT_RXNE);
				//1、退出透传 +++
				sprintf(response_buff, "+++");
				for (i=0;i<2;i++)
				{
					HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, 3);
					HAL_Delay(200);
				}
				//2、开启小程序配网：AT+SMARTSTART=6\r\n
				sprintf(response_buff, "AT+SMARTSTART=6\r\n");
				for (i=0;i<2;i++)
				{
					HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));
					HAL_Delay(200);
				}
				//3、设置模块做 UDP 服务器端 的参数
				sprintf(response_buff, "AT+UARTFOMAT=68,50\r\n");
				for (i=0;i<2;i++)
				{
					HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));
					HAL_Delay(100);
				}
				//4、透传模式：AT+CIPSENDRAW
				sprintf(response_buff, "AT+CIPSENDRAW\r\n");
				for (i=0;i<2;i++)
				{
				HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, 15);
				HAL_Delay(100);
				}
				__HAL_UART_ENABLE_IT(&huart1,UART_IT_RXNE);
				break;
				
			
			case 0x0011:
				main_state = 0x1111;
				HAL_Delay(100);
				break;
			
			/* 触发数据发送 */
			case 0x1111:
				Send_SycMSG.MSGHead.length = __REV16(sizeof(Send_SycMSG)-8); //帧长度
				if(AD1_NUM>=1024){
					HAL_NVIC_DisableIRQ(EXTI15_10_IRQn);
					for(i = 0; i < 8; i++){
						arm_rfft_f32_32T(0, i, AD1_buffer[i]); // AD, CH
					}
					//发送数据+采集初始化
					Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
					HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, sizeof(Send_SycMSG));
					
					
					AD1_NUM=0;
					AD2_NUM=0;
					HAL_Delay(20);
					HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);
					//HAL_GPIO_TogglePin(AD9833_CS3_GPIO_Port, AD9833_CS3_Pin);
					sent_num_flag = 0;
				}
				break;
			
			/* 触发校准回复 */
			case 0x0003:
				main_state = 0x0303;
				//HAL_Delay(100);
				break;
			
			case 0x0303:
				Send_SycMSG.MSGHead.length = __REV16(sizeof(Send_SycMSG)-8); //帧长度
				if(AD1_NUM>=1024){
					HAL_NVIC_DisableIRQ(EXTI15_10_IRQn);
					for(i = 0; i < 8; i++){
						arm_rfft_f32_32T(0, i, AD1_buffer[i]); // AD, CH
					}
					//发送数据+采集初始化
					Send_SycMSG.MSGHead.sequentialNUM = __REV16(__REV16(Send_SycMSG.MSGHead.sequentialNUM)+1); //帧数递增
					HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Send_SycMSG, sizeof(Send_SycMSG));
					
					
					AD1_NUM=0;
					AD2_NUM=0;
					HAL_Delay(20);
					HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);
					//HAL_GPIO_TogglePin(AD9833_CS3_GPIO_Port, AD9833_CS3_Pin);
					sent_num_flag = 0;
				}
				break;
				
			case 0x0404:
				
				if(AD1_NUM>=1024 && AD2_NUM>=1024){
					HAL_NVIC_DisableIRQ(EXTI15_10_IRQn);
					
					//arm_rfft_f32_32T(0, 0, AD2_buffer[0]); // AD, CH

					for(i = 0; i < 8; i++){
						arm_rfft_f32_32T(0, 0, AD1_buffer[i]); // AD, CH
					}
					
					for(i = 0; i < 8; i++){
						arm_rfft_f32_32T(0, 0, AD2_buffer[i]); // AD, CH
					}
					
					CDC_Transmit_HS(sentdata_16x16, sizeof(sentdata_16x16));


//					for(i = 0; i < 1024; i++){
//					VOFA_Float[0] = (float)AD2_buffer[0][i];
//					VOFA_Float[1] = (float)AD1_buffer[0][i];
//					VOFA_Float[2] = (float)111.11;//ADC_DMA_ConvertedValue[i];
//					VOFA_Float[3] = (float)111.11;//ADC_DMA_ConvertedValue[i];
//					memcpy(VOFA_Data,(uint8_t*)VOFA_Float,16);
//					CDC_Transmit_HS(VOFA_Data, sizeof(VOFA_Data));
//					for_delay_us(10);
//					}

					HAL_GPIO_TogglePin(LED_KEY_GPIO_Port, LED_KEY_Pin);
          AD1_NUM=0;
					AD2_NUM=0;
					//HAL_Delay(100);
					HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

					sent_num_flag = 0;
				}
				break;	
				
		  default:
				break;
		}



		
			
		
}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48|RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = 64;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 34;
  RCC_OscInitStruct.PLL.PLLP = 1;
  RCC_OscInitStruct.PLL.PLLQ = 5;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 3072;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief NVIC Configuration.
  * @retval None
  */
static void MX_NVIC_Init(void)
{
  /* EXTI15_10_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);
}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_HIGH;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_64;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 0x0;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  hspi1.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
  hspi1.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
  hspi1.Init.TxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi1.Init.RxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi1.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
  hspi1.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
  hspi1.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
  hspi1.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
  hspi1.Init.IOSwap = SPI_IO_SWAP_DISABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 5-1;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 275-1;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 137;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 921600;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/* FMC initialization function */
static void MX_FMC_Init(void)
{

  /* USER CODE BEGIN FMC_Init 0 */

  /* USER CODE END FMC_Init 0 */

  FMC_NORSRAM_TimingTypeDef Timing = {0};

  /* USER CODE BEGIN FMC_Init 1 */

  /* USER CODE END FMC_Init 1 */

  /** Perform the NOR1 memory initialization sequence
  */
  hnor1.Instance = FMC_NORSRAM_DEVICE;
  hnor1.Extended = FMC_NORSRAM_EXTENDED_DEVICE;
  /* hnor1.Init */
  hnor1.Init.NSBank = FMC_NORSRAM_BANK1;
  hnor1.Init.DataAddressMux = FMC_DATA_ADDRESS_MUX_ENABLE;
  hnor1.Init.MemoryType = FMC_MEMORY_TYPE_NOR;
  hnor1.Init.MemoryDataWidth = FMC_NORSRAM_MEM_BUS_WIDTH_16;
  hnor1.Init.BurstAccessMode = FMC_BURST_ACCESS_MODE_DISABLE;
  hnor1.Init.WaitSignalPolarity = FMC_WAIT_SIGNAL_POLARITY_LOW;
  hnor1.Init.WaitSignalActive = FMC_WAIT_TIMING_BEFORE_WS;
  hnor1.Init.WriteOperation = FMC_WRITE_OPERATION_DISABLE;
  hnor1.Init.WaitSignal = FMC_WAIT_SIGNAL_DISABLE;
  hnor1.Init.ExtendedMode = FMC_EXTENDED_MODE_DISABLE;
  hnor1.Init.AsynchronousWait = FMC_ASYNCHRONOUS_WAIT_DISABLE;
  hnor1.Init.WriteBurst = FMC_WRITE_BURST_DISABLE;
  hnor1.Init.ContinuousClock = FMC_CONTINUOUS_CLOCK_SYNC_ONLY;
  hnor1.Init.WriteFifo = FMC_WRITE_FIFO_ENABLE;
  hnor1.Init.PageSize = FMC_PAGE_SIZE_NONE;
  /* Timing */
  Timing.AddressSetupTime = 6;
  Timing.AddressHoldTime = 3;
  Timing.DataSetupTime = 8;
  Timing.BusTurnAroundDuration = 0;
  Timing.CLKDivision = 16;
  Timing.DataLatency = 17;
  Timing.AccessMode = FMC_ACCESS_MODE_A;
  /* ExtTiming */

  if (HAL_NOR_Init(&hnor1, &Timing, NULL) != HAL_OK)
  {
    Error_Handler( );
  }

  /** Perform the NOR2 memory initialization sequence
  */
  hnor2.Instance = FMC_NORSRAM_DEVICE;
  hnor2.Extended = FMC_NORSRAM_EXTENDED_DEVICE;
  /* hnor2.Init */
  hnor2.Init.NSBank = FMC_NORSRAM_BANK2;
  hnor2.Init.DataAddressMux = FMC_DATA_ADDRESS_MUX_ENABLE;
  hnor2.Init.MemoryType = FMC_MEMORY_TYPE_NOR;
  hnor2.Init.MemoryDataWidth = FMC_NORSRAM_MEM_BUS_WIDTH_16;
  hnor2.Init.BurstAccessMode = FMC_BURST_ACCESS_MODE_DISABLE;
  hnor2.Init.WaitSignalPolarity = FMC_WAIT_SIGNAL_POLARITY_LOW;
  hnor2.Init.WaitSignalActive = FMC_WAIT_TIMING_BEFORE_WS;
  hnor2.Init.WriteOperation = FMC_WRITE_OPERATION_DISABLE;
  hnor2.Init.WaitSignal = FMC_WAIT_SIGNAL_DISABLE;
  hnor2.Init.ExtendedMode = FMC_EXTENDED_MODE_DISABLE;
  hnor2.Init.AsynchronousWait = FMC_ASYNCHRONOUS_WAIT_DISABLE;
  hnor2.Init.WriteBurst = FMC_WRITE_BURST_DISABLE;
  hnor2.Init.ContinuousClock = FMC_CONTINUOUS_CLOCK_SYNC_ONLY;
  hnor2.Init.WriteFifo = FMC_WRITE_FIFO_ENABLE;
  hnor2.Init.PageSize = FMC_PAGE_SIZE_NONE;
  /* Timing */
  Timing.AddressSetupTime = 6;
  Timing.AddressHoldTime = 3;
  Timing.DataSetupTime = 8;
  Timing.BusTurnAroundDuration = 0;
  Timing.CLKDivision = 16;
  Timing.DataLatency = 17;
  Timing.AccessMode = FMC_ACCESS_MODE_A;
  /* ExtTiming */

  if (HAL_NOR_Init(&hnor2, &Timing, NULL) != HAL_OK)
  {
    Error_Handler( );
  }

  /* USER CODE BEGIN FMC_Init 2 */

  /* USER CODE END FMC_Init 2 */
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, LED_KEY_Pin|AD_RESET_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, MUX_A3_Pin|MUX_A2_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, MUX_A1_Pin|MUX_A0_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(MUX_CS_GPIO_Port, MUX_CS_Pin, GPIO_PIN_SET);

  /*Configure GPIO pins : LED_KEY_Pin AD_RESET_Pin */
  GPIO_InitStruct.Pin = LED_KEY_Pin|AD_RESET_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : KEY1_Pin */
  GPIO_InitStruct.Pin = KEY1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(KEY1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : KEY2_Pin */
  GPIO_InitStruct.Pin = KEY2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(KEY2_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : MUX_A3_Pin MUX_A2_Pin */
  GPIO_InitStruct.Pin = MUX_A3_Pin|MUX_A2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : MUX_A1_Pin MUX_A0_Pin */
  GPIO_InitStruct.Pin = MUX_A1_Pin|MUX_A0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : MUX_CS_Pin */
  GPIO_InitStruct.Pin = MUX_CS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(MUX_CS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : BUSY1_Pin BUSY2_Pin */
  GPIO_InitStruct.Pin = BUSY1_Pin|BUSY2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI3_IRQn);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */



/**
  * @brief  This function writes 2 bytes to AD9833(SPI1).
  * @param Data to transmit
  * @retval None
  */
void AD9833_Write(SPI_HandleTypeDef* spi_handle, unsigned short TxData)
{
	unsigned char data[2];
	data[0] = (unsigned char)((TxData>>8)&0xff);
	data[1] = (unsigned char)(TxData&0xff);
	HAL_SPI_Transmit(spi_handle, data, 2, 1);
}

/**
  * @brief  This function configures AD9833
	* @param reset: No output when 1. Does not reset regisiters.
	* @param sleepMode: 0 normal output
	*                   1 shutdown DAC
	*                   2 shutdown internal clock(pause output)
	*                   3 shutdown DAC and internal clock
	* @param waveform: 0 sine wave
	*                   1 triangle wave
	*                   2 ramp wave
	*                   3 reserved
  * @retval None
  */
void AD9833_Config(SPI_HandleTypeDef* spi_handle, unsigned char reset, unsigned char sleepMode, unsigned char waveform)
{
	unsigned short reg16 = 0;
	reg16 |= ((unsigned short) reset & 0x01) << 8;
	reg16 |= (sleepMode & 0x03) << 6;
	reg16 |= (waveform & 0x02) << 4;
	reg16 |= (waveform & 0x01) << 1;
	AD9833_Write(spi_handle, 0x2000 | reg16);
}

/**
  * @brief  This function sets output freq of AD9833
	* @param freq: output freq(Hz)
  * @retval None
  */
void AD9833_SetFreq(SPI_HandleTypeDef* spi_handle, double freq)
{
	long freq_bin;
	int freq_LSB, freq_MSB;
	freq_bin = freq / 25000000 * (1L << 28) + 0.5; //round to neareset whole number
	freq_LSB = freq_bin & 0x3fff;
	freq_MSB = (freq_bin >> 14) & 0x3fff;
	AD9833_Write(spi_handle, 0x2100);
	AD9833_Write(spi_handle, 0x4000 | freq_LSB);
	AD9833_Write(spi_handle, 0x4000 | freq_MSB);
}



static int arm_rfft_f32_32T(uint8_t ad_label, uint8_t ad_channel, float32_t* ad_array)
{

	char transStr[60];
	int16_t result_data;
	int16_t i, num;
	float32_t target_data,target_data_out;
	float32_t lX,lY;
	arm_rfft_fast_instance_f32 S;
	/* 正变换 */
	uint8_t ifftFlag = 0;

	/* 初始化结构体S中的参数 */
	arm_rfft_fast_init_f32(&S, FFT_LENGTH);
	/* FFT_LENGTH 点实序列快速 FFT */
	arm_rfft_fast_f32(&S, ad_array, testOutput_f32, ifftFlag);
	/* 为了方便跟函数 arm_cfft_f32 计算的结果做对比，这里求解了 FFT_LENGTH 组模值，实际函数 arm_rfft_fast_f32
	只求解出了 512 组*/
	//arm_cmplx_mag_f32(testOutput_f32, testOutputMag_f32, FFT_LENGTH);


//	for (i =0; i < FFT_LENGTH / 2; i++)
//	{
//		lX = testOutput_f32[2*i]; /* 实部*/
//		lY = testOutput_f32[2*i+1]; /* 虚部 */
//		testOutputMag_f32[i] = sqrt(lX*lX+ lY*lY); /* 求模 */
//		testOutputMag_f32[i] = testOutputMag_f32[i]/FFT_LENGTH/2;
//		
//    sentdata_decodeall[i+1] = (int)testOutputMag_f32[i];
//	}
//	
//	CDC_Transmit_HS((uint8_t*)sentdata_decodeall, 514*2);
	
	
	
	for (int fre =0; fre < 16; fre++)      //for (i =103; i < 183; i+=5)
	{
		//-1
		i = (int)(5.12*(fre+20));
		num = i-1;
		lX = testOutput_f32[num<<1]; /* 实部*/
		lY = testOutput_f32[num<<1|1]; /* 虚部 */
		testOutputMag_f32[num] = lX*lX+lY*lY; /* 求模 */

		//0
		num = i;
		lX = testOutput_f32[num<<1]; /* 实部*/
		lY = testOutput_f32[num<<1|1]; /* 虚部 */
		testOutputMag_f32[num] = lX*lX+lY*lY; /* 求模 */

		//1
		num = i + 1;
		lX = testOutput_f32[num<<1]; /* 实部*/
		lY = testOutput_f32[num<<1|1]; /* 虚部 */
		testOutputMag_f32[num] = lX*lX+lY*lY; /* 求模 */

		target_data = max(max(testOutputMag_f32[i-1],testOutputMag_f32[i]),testOutputMag_f32[i+1]);
		arm_sqrt_f32(target_data,&target_data_out);

		target_data = target_data_out/FFT_LENGTH/206;//(long long) sqrt(target_data) >> (FFT_LENGTH_EXP - 1);
		target_data = (target_data * i ); // LENTH/2 / 103


		//target_data = 5555.55555;
		sentdata_16x16[++sent_num_flag] = (int) target_data;
		
//		//放入整数字段
//		Send_SycMSG.MSGbody[sent_num_flag-1].integer_Data[2] = (int) target_data&0XFF;//低位 0-8
//		Send_SycMSG.MSGbody[sent_num_flag-1].integer_Data[1] = (int) target_data >> 8;//8-16
//		Send_SycMSG.MSGbody[sent_num_flag-1].integer_Data[0] = (int) target_data >> 16;//高位 16-24
//		
//		Send_SycMSG.MSGbody[sent_num_flag-1].decimal_Data[2] = (int) (target_data*1000)&0XFF;//低位 0-8
//		Send_SycMSG.MSGbody[sent_num_flag-1].decimal_Data[1] = (int) (target_data*1000) >> 8;//8-16
//		Send_SycMSG.MSGbody[sent_num_flag-1].decimal_Data[0] = (int) (target_data*1000) >> 16;//高位 16-24
	}
}



//USB 接口 复位
void USB_Close(void)
{
  GPIO_InitTypeDef GPIO_InitStruct;
  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_12, GPIO_PIN_RESET);
  /*Configure GPIO pin : PA12 */
  GPIO_InitStruct.Pin = GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
	
  //先把PA12拉低再拉高，利用D+模拟USB的拔插动作
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_12, GPIO_PIN_RESET);
  HAL_Delay(65);
  HAL_GPIO_WritePin(GPIOA,GPIO_PIN_12,GPIO_PIN_SET);
  HAL_Delay(65);
}






//中断读取ADC
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	static int16_t i = 0;
	static int16_t AD_data;
	static int16_t oldValue_AD1 = 0;
	static int16_t oldValue_AD2 = 0;
	static int16_t DETA_limit = 1000;
	
	//AD1
	if(GPIO_Pin == GPIO_PIN_11){
		//读取ADC数据
		if (AD1_NUM < FFT_LENGTH){
			for (i=0;i<ADC_CHANNEL;i++)
			{
				AD_data = AD7606_RESULT_1();
        AD1_buffer[i][AD1_NUM] = AD_data * window_func[AD1_NUM] *10;

			}
			AD1_NUM++;
		}
	}
	
	//AD2
	if(GPIO_Pin == GPIO_PIN_12){
		//读取ADC数据
		if (AD2_NUM < FFT_LENGTH){
			for (i=0;i<ADC_CHANNEL;i++)
			{
				AD_data = AD7606_RESULT_2();
				AD2_buffer[i][AD2_NUM] = AD_data * window_func[AD2_NUM] *10;
			}
			AD2_NUM++;
		}

  }
	
	
	
	//按键1KEY1
		if(GPIO_Pin == KEY1_Pin){
			if(HAL_GPIO_ReadPin(KEY1_GPIO_Port,KEY1_Pin)== 0)
			{
//				sprintf(response_buff, "KEY1_Matching network\r\n");
//			  HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));		
        /* 进入配网状态 */				
				main_state = 1;
			}
	}
	
	
}


void Init_window_func(float32_t* win_func, int length, int type)
//type 0:矩形窗 1:三角窗 2:hanning窗 3:hamming窗 4:blackman窗
{
	int i;
	switch(type)
	{
		case 0://矩形窗
			for(i = 0; i < length; ++i)
				win_func[i] = 1;
			break;
		case 1://三角窗
			for(i = 0; i < (length >> 1); ++i)
				win_func[i] = ((i << 1) - 1) / (float32_t) length;
			for(i = (length >> 1); i < length; ++i)
				win_func[i] = 2 - ((i << 1) - 1) / (float32_t) length;
			break;
		case 2://hanning窗
			for(i = 0; i < length; ++i)
				win_func[i] = 0.5 - 0.5 * arm_cos_f32(2 * PI * i / (length - 1));
			break;
		case 3://hamming窗
			for(i = 0; i < length; ++i)
				win_func[i] = 0.54 - 0.46 * arm_cos_f32(2 * PI * i / (length - 1));
			break;
		case 4://blackman窗
			for(i = 0; i < (length >> 1); ++i)
				win_func[i] = win_func[length - i -1] = 0.42 - 0.5 * arm_cos_f32(2 * PI * i / (length - 1)) + 0.08 * arm_cos_f32(4 * PI * i / (length - 1));
			break;
	}
}



//串口中断接收
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	char tx_data2[] = "receive 0!\r\n"; 
	char tx_data3[] = "receive 1!\r\n"; 
	int i;
	int cnt_index;
	
	
	
	if(offset_index >= 200){ //以防溢出
		offset_index = 0;
	}	
	
	
  if (huart->Instance == USART1) //
  {
		rxbuff[offset_index++] = uart1_rxdata[0]; 
    HAL_UART_Receive_IT(&huart1, (uint8_t*)uart1_rxdata, sizeof(uart1_rxdata)); //
  }
	
	
	if(rxbuff[offset_index-1] == '\n' && rxbuff[offset_index-2] == '\r' && offset_index>20)
	    {
			rxbuff[offset_index-1] = '\0';//将\n字符清零
			rxbuff[offset_index-2] = '\0';//将\r字符清零 用于字符串判断	
			cnt_index = 0;
			for (i=0;i<20;i++)
			{		
			 hand_buff[i] = rxbuff[offset_index-2-20+i];
			}
			memcpy(&Get_SycMSG, hand_buff, 20);
			
			//HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Get_SycMSG, 20);
			//HAL_UART_Transmit_IT(&huart1, (uint8_t*)&Get_SycMSG.MSGHead.CMD, sizeof(Get_SycMSG.MSGHead.CMD));
      State_CMD();
			offset_index = 0;//数组下标放到最前面
			return ;//接收结束直接返回
		}
	
}



/* WIFI初始化 */
static void MX3090_Init(void)
{	
	int i;

	/* 921600 */
	//1、退出透传 +++
	memset(&response_buff, 0, sizeof(response_buff));
	sprintf(response_buff, "+++");
	for (i=0;i<2;i++)
	{
		HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, 3);
		HAL_Delay(200);
	}
	//2、出厂设置
	memset(&response_buff, 0, sizeof(response_buff));
	sprintf(response_buff, "AT+FACTORY\r\n");
	for (i=0;i<4;i++)
	{
		HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));
		HAL_Delay(100);
	}
	
	/* 115200 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }

	//1、退出透传 +++
	memset(&response_buff, 0, sizeof(response_buff));
	sprintf(response_buff, "+++");
	for (i=0;i<2;i++)
	{
		HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, 3);
		HAL_Delay(200);
	}
	//2、出厂设置
	memset(&response_buff, 0, sizeof(response_buff));
	sprintf(response_buff, "AT+FACTORY\r\n");
	for (i=0;i<4;i++)
	{
		HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));
		HAL_Delay(200);
	}
	//3、改波特率
	sprintf(response_buff, "AT+UART=921600,8,1,NONE,NONE\r\n");
	for (i=0;i<2;i++)
	{
		HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));
		HAL_Delay(200);
	}
	
	/* 921600 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 921600;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
	
	//1、AP模式
	sprintf(response_buff, "AT+WSAP=zBean_1,12345678\r\n");
	for (i=0;i<2;i++)
	{
		HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));
		HAL_Delay(200);
	}
	
	//2、广播
	sprintf(response_buff, "AT+CIPSTART=2,udp_broadcast,192.168.100.255,8091,8061\r\n");
	for (i=0;i<2;i++)
	{
		HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));
		HAL_Delay(200);
	}
	
	//3、透传
	sprintf(response_buff, "AT+CIPSENDRAW\r\n");
	for (i=0;i<2;i++)
	{
		HAL_UART_Transmit_IT(&huart1, (uint8_t*)response_buff, sizeof(response_buff));
		HAL_Delay(200);
	}
}

void MUX_4_16(uint8_t channel)
//4-16译码器 channel:要显示的管编号 0-15(共 16 个数码管)
{
	HAL_GPIO_WritePin(MUX_A0_GPIO_Port,MUX_A0_Pin, (channel&0x01) ? GPIO_PIN_SET : GPIO_PIN_RESET);
	HAL_GPIO_WritePin(MUX_A1_GPIO_Port,MUX_A1_Pin, ((channel&0x02)>>1) ? GPIO_PIN_SET : GPIO_PIN_RESET);
	HAL_GPIO_WritePin(MUX_A2_GPIO_Port,MUX_A2_Pin, ((channel&0x04)>>2) ? GPIO_PIN_SET : GPIO_PIN_RESET);
	HAL_GPIO_WritePin(LED_KEY_GPIO_Port,LED_KEY_Pin, ((channel&0x08)>>3) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

void for_delay_us(uint32_t nus)
{
    uint32_t Delay = nus * 168/4;
    do
    {
        __NOP();
    }
    while (Delay --);
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
