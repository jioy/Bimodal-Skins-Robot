/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define LED_KEY_Pin GPIO_PIN_14
#define LED_KEY_GPIO_Port GPIOC
#define KEY1_Pin GPIO_PIN_13
#define KEY1_GPIO_Port GPIOC
#define KEY1_EXTI_IRQn EXTI15_10_IRQn
#define KEY2_Pin GPIO_PIN_3
#define KEY2_GPIO_Port GPIOB
#define KEY2_EXTI_IRQn EXTI3_IRQn
#define MUX_A3_Pin GPIO_PIN_7
#define MUX_A3_GPIO_Port GPIOC
#define MUX_A2_Pin GPIO_PIN_6
#define MUX_A2_GPIO_Port GPIOC
#define AD_RESET_Pin GPIO_PIN_4
#define AD_RESET_GPIO_Port GPIOC
#define MUX_A1_Pin GPIO_PIN_15
#define MUX_A1_GPIO_Port GPIOB
#define MUX_A0_Pin GPIO_PIN_14
#define MUX_A0_GPIO_Port GPIOB
#define MUX_CS_Pin GPIO_PIN_13
#define MUX_CS_GPIO_Port GPIOD
#define BUSY1_Pin GPIO_PIN_11
#define BUSY1_GPIO_Port GPIOB
#define BUSY1_EXTI_IRQn EXTI15_10_IRQn
#define BUSY2_Pin GPIO_PIN_12
#define BUSY2_GPIO_Port GPIOB
#define BUSY2_EXTI_IRQn EXTI15_10_IRQn

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
