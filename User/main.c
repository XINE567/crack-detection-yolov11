#include "stm32f10x.h"
#include "Delay.h"
#include "LED.h"
#include "NeZha.h"
#include "sys.h"
#include "ps2.h"
#include "Usart.h"
#include "RobotArm.h"
#include "Board_Timer.h"
#include "Vehicle_Chassis.h"

#define   UNIT_PWM	1				//舵机单位转动值	
#define   PS2_LSPEED    1000		//左遥杆拨到底对应电机最大目标值（电机最大速度为1000）
#define   PS2_RSPEED    800			//右遥杆拨到底对应电机最大目标值（电机最大速度为800）
#define   RECIPROCAL	0.0078f     //128的倒数，此处为方便摇杆值映射目标值的计算。不可更改

uint8_t KeyNum;
uint8_t PS2_Mode,Last_PS2_Mode;
int16_t M1_Target, M2_Target, M3_Target, M4_Target;
uint8_t Time;
uint8_t Led_State;  //车尾灯灯状态标志位； 0：熄灭， 1：后退， 2：左转弯灯， 3：右转弯灯

int main(void)
{
 	__disable_irq(); 
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	
	LED_Init();	
	PS2_Init();
	NeZha_Init();
	RobotArm_Init(); 
	Vehicle_Chassis_Init();
	Board_Timer_Init();
	__enable_irq();  
	while (1)
	{
		if (Board_Timer_Flag_Get())    //定时器定时5ms周期
		{
			Time++;

			if ((Time + 1)%2 == 0)	//遥控控制周期10ms
			{
				KeyNum = ps2_key_serch();
				PS2_Mode = ps2_mode_get();				
				
				/**************小车底盘控制*****************/							
				if(PS2_Mode == PSB_REDLIGHT_MODE)
				{
					static unsigned char ps2_ly,ps2_rx,ps2_ry;
					ps2_ly = ps2_get_anolog_data(PSS_LY);
					ps2_rx = ps2_get_anolog_data(PSS_RX);	
					ps2_ry = ps2_get_anolog_data(PSS_RY);	

					if (ps2_get_key_state(PSB_L1))  //机械臂控制
					{
						if (ps2_ly == 0x00)	RobotArm_StretchHand(UNIT_PWM);	
						if (ps2_ly == 0xff) RobotArm_ShrinkHand(UNIT_PWM);		
						if (ps2_ry == 0x00)	RobotArm_DropHand(UNIT_PWM);	
						if (ps2_ry == 0xff) RobotArm_RaiseHand(UNIT_PWM);							
						if (ps2_rx == 0x00)	RobotArm_LetHand(UNIT_PWM);
						if (ps2_rx == 0xff) RobotArm_ShakeHand(UNIT_PWM);		
						//速度清零
						M1_Target = 0;
						M4_Target = 0;
						M2_Target = 0;
						M3_Target = 0;
						
						//关闭转弯灯
						Led_State = 0;
					}
					else   //小车底盘控制控制
					{
						//遥感数据处理
						int rx_adjust,ly_adjust;
						int ly_target = 0;

						rx_adjust = ps2_rx - 128;
						ly_adjust = ps2_ly - 127;
		
						rx_adjust = (unsigned int)(ps2_rx*0.235 + 120);    //右摇杆X轴映射舵机角度
						Arc_ServoPwm_Set(rx_adjust);
						
						ly_target = PS2_LSPEED  * ly_adjust * RECIPROCAL;  //左摇杆Y轴映射电机速度

					    //尾灯状态判断； 0：熄灭， 1：后退， 2：左转弯灯， 3：右转弯灯
						if(ly_target > 0)    //倒车
						{
						  Led_State = 1;
						}
						else 
						{
						  if (rx_adjust < 140)  //左转弯灯
						  {
							Led_State = 2;
						  }
						  else if(rx_adjust > 160) //右转弯灯
						  {
							 Led_State = 3;
						  }
						  else
						  {
							 Led_State = 0;
						  }
						}
						
						//各电机转速计算						
						M1_Target =  ly_target;
						M2_Target = -ly_target; 
						M3_Target = -ly_target;
						M4_Target =  ly_target;	
					}
				}			
				else if(PS2_Mode == PSB_GREENLIGHT_MODE)
				{
					if (ps2_get_key_state(PSB_L1))  //机械臂控制
					{
						if (ps2_get_key_state(PSB_PAD_UP))	   RobotArm_StretchHand(UNIT_PWM);	  
						if (ps2_get_key_state(PSB_PAD_DOWN))   RobotArm_ShrinkHand(UNIT_PWM);   
						if (ps2_get_key_state(PSB_GREEN))	   RobotArm_DropHand(UNIT_PWM);	
						if (ps2_get_key_state(PSB_BLUE))       RobotArm_RaiseHand(UNIT_PWM);								
						if (ps2_get_key_state(PSB_PINK))	   RobotArm_LetHand(UNIT_PWM);
						if (ps2_get_key_state(PSB_RED))		   RobotArm_ShakeHand(UNIT_PWM);			
					
						//速度清零
						M1_Target = 0;
						M4_Target = 0;
						M2_Target = 0;
						M3_Target = 0;	
					}
					else   //小车底盘控制控制
					{	
						unsigned char up_state,down_state;
						unsigned char pink_state,red_state;
						int ly_target = 0;

						up_state = ps2_get_key_state(PSB_PAD_UP);
						down_state = ps2_get_key_state(PSB_PAD_DOWN);
						pink_state = ps2_get_key_state(PSB_PINK);
						red_state = ps2_get_key_state(PSB_RED);

						
						if (pink_state) Arc_ServoPwm_TurnLeft(UNIT_PWM);
						else if (red_state) Arc_ServoPwm_TurnRight(UNIT_PWM);

						up_state == 1?   ly_target = PS2_LSPEED : (down_state == 1? (ly_target = 0 - PS2_LSPEED) : (ly_target = 0));

						//尾灯状态判断； 0：熄灭， 1：后退， 2：左转弯灯， 3：右转弯灯
						if(ly_target < 0)    //倒车
						{
							Led_State = 1;
						}
						else 
						{
							if (pink_state)  //左转弯灯
							{
								Led_State = 2;
							}
							else if(red_state) //右转弯灯
							{
								Led_State = 3;
							}
							else
							{
								Led_State = 0;
							}
						}
						
						M1_Target = -ly_target;
						M2_Target =  ly_target; 
						M3_Target =  ly_target;
						M4_Target = -ly_target;	
					}
				}
				Last_PS2_Mode = PS2_Mode;
			}		
			if ((Time + 1)%5 == 0)       //电机控制  25ms
			{
				M1_Target >= 0? NeZha_Motor1_SetPwm(0,M1_Target) : NeZha_Motor1_SetPwm(0 - M1_Target,0);
				M2_Target >= 0? NeZha_Motor2_SetPwm(0,M2_Target) : NeZha_Motor2_SetPwm(0 - M2_Target,0);
				M3_Target >= 0? NeZha_Motor3_SetPwm(0,M3_Target) : NeZha_Motor3_SetPwm(0 - M3_Target,0);
				M4_Target >= 0? NeZha_Motor4_SetPwm(0,M4_Target) : NeZha_Motor4_SetPwm(0 - M4_Target,0);
			}
			if ((Time + 1)%50 == 0)   //尾灯控制  250ms
			{
				//车尾灯控制； 0：熄灭， 1：后退， 2：左转弯灯， 3：右转弯灯
				switch(Led_State)
				{
				  case 0:{
					NeZha_TailLeftLed_TurnOff();
					NeZha_TailRightLed_TurnOff();
				  }break;
				  case 1:{
					NeZha_TailLeftLed_TurnOn();
					NeZha_TailRightLed_TurnOn();
				  }break;
				  case 2:{
					NeZha_TailLeftLed_Turn();
					NeZha_TailRightLed_TurnOff();
				  }break;     
				  case 3:{
					NeZha_TailLeftLed_TurnOff();
					NeZha_TailRightLed_Turn();
				  }break;  
				  default:{
					NeZha_TailLeftLed_TurnOff();
					NeZha_TailRightLed_TurnOff();
				  }break;     
				}
			}
			if((Time + 1)%200 == 0)		//指示灯   1s闪烁
			{
				LED1_Turn();
				Time = 0;
			}
		}	
	}
}

