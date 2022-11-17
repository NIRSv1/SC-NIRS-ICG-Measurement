#i f ö.,öz o,  4# -*- coding: utf-8 -*-
"""
@author: hmchayme
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import random
import serial
import sys
import csv
import keyboard
import time
from IPython.display import clear_output
from datetime import datetime


#initialize serial port
ser = serial.Serial()
ser.port = 'COM8' #Arduino serial port
ser.baudrate = 112500
ser.open()

#LED Wavelength lambda in nm
lambda_1=690
lambda_2=780
lambda_3=810
lambda_4=850

#transport scattering coefficient µs'=b*lambda^(-a) in (mm^-1) @matcher(1997) 
a_forearm=0.576
b_forearm=32.48
µs_forearm_690=b_forearm*lambda_1**(-a_forearm)
µs_forearm_780=b_forearm*lambda_2**(-a_forearm)
µs_forearm_810=b_forearm*lambda_3**(-a_forearm)
µs_forearm_850=b_forearm*lambda_4**(-a_forearm)


µs_forearm_690_780_810_850=np.array([µs_forearm_690,µs_forearm_780,µs_forearm_810,µs_forearm_850])

# epsylon_lambda_Chro:  Extinction coefficient in (mM^-1* mm^-1)
epsylon_690_HbO=0.03123
epsylon_690_HbR=0.21382
epsylon_690_CtOx=0.263213
epsylon_690_ICG=6.48732

epsylon_780_HbO=0.073583
epsylon_780_HbR=0.110471
epsylon_780_CtOx=0.204948
epsylon_780_ICG=26.50983

epsylon_810_HbO=0.092887
epsylon_810_HbR=0.079849
epsylon_810_CtOx=0.231661
epsylon_810_ICG=43.92281

epsylon_850_HbO=0.11531
epsylon_850_HbR=0.07861
epsylon_850_CtOx=0.22899
epsylon_850_ICG=6.82102

#epsylon_lambda_chromophore=[   epsylon_690_HbO epsylon_690_HbR epsylon_690_CtOx  epsylon_690_ICG
#                               epsylon_780_HbO epsylon_780_HbR epsylon_780_CtOx  epsylon_780_ICG
#                               epsylon_810_HbO epsylon_810_HbR epsylon_810_CtOx  epsylon_810_ICG
#                               epsylon_850_HbO epsylon_850_HbR epsylon_850_CtOx  epsylon_850_ICG]


epsylon_690_780_810_850_HbO_HbR_CtOx_ICG=np.array([[epsylon_690_HbO, epsylon_690_HbR, epsylon_690_CtOx,  epsylon_690_ICG],
                                                          [epsylon_780_HbO, epsylon_780_HbR, epsylon_780_CtOx,  epsylon_780_ICG],
                                                          [epsylon_810_HbO, epsylon_810_HbR, epsylon_810_CtOx,  epsylon_810_ICG],
                                                          [epsylon_850_HbO, epsylon_850_HbR, epsylon_850_CtOx,  epsylon_850_ICG]])
A=[]
B=np.empty([1,128])
B1=np.empty([1,64])
B2=np.empty([1,64])
Abs_Concentration_690_780_810_850_HbO_HbR_CtOx_ICG_distance_1_2_3_4_5_6_iter=np.empty([1,64])
RowVoltage ="D:/NIRS/Python/NIRS-Sensor/Data/Row-Voltage"+ time.strftime("%Y%m%d-%H%M%S")+".csv"


 
with open(RowVoltage,'a') as csvfile:
    np.savetxt(csvfile, A,delimiter=',',header='TStamp, Trg, Frame,\
LED_1_PD_1, LED_1_PD_2, LED_1_PD_3, LED_1_PD_4, LED_1_PD_5, LED_1_PD_6, LED_1_PD_7, LED_1_PD_8,\
LED_2_PD_1, LED_2_PD_2, LED_2_PD_3, LED_2_PD_4, LED_2_PD_5, LED_2_PD_6, LED_2_PD_7, LED_2_PD_8,\
LED_3_PD_1, LED_3_PD_2, LED_3_PD_3, LED_3_PD_4, LED_3_PD_5, LED_3_PD_6, LED_3_PD_7, LED_3_PD_8,\
LED_4_PD_1, LED_4_PD_2, LED_4_PD_3, LED_4_PD_4, LED_4_PD_5, LED_4_PD_6, LED_4_PD_7, LED_4_PD_8,\
LED_5_PD_1, LED_5_PD_2, LED_5_PD_3, LED_5_PD_4, LED_5_PD_5, LED_5_PD_6, LED_5_PD_7, LED_5_PD_8,\
LED_6_PD_1, LED_6_PD_2, LED_6_PD_3, LED_6_PD_4, LED_6_PD_5, LED_6_PD_6, LED_6_PD_7, LED_6_PD_8,\
LED_7_PD_1, LED_7_PD_2, LED_7_PD_3, LED_7_PD_4, LED_7_PD_5, LED_7_PD_6, LED_7_PD_7, LED_7_PD_8,\
LED_8_PD_1, LED_8_PD_2, LED_8_PD_3, LED_8_PD_4, LED_8_PD_5, LED_8_PD_6, LED_8_PD_7, LED_8_PD_8,\
LED_9_PD_1, LED_9_PD_2, LED_9_PD_3, LED_9_PD_4, LED_9_PD_5, LED_9_PD_6, LED_9_PD_7, LED_9_PD_8,\
LED_10_PD_1, LED_10_PD_2, LED_10_PD_3, LED_10_PD_4, LED_10_PD_5, LED_10_PD_6, LED_10_PD_7, LED_10_PD_8,\
LED_11_PD_1, LED_11_PD_2, LED_11_PD_3, LED_11_PD_4, LED_11_PD_5, LED_11_PD_6, LED_11_PD_7, LED_11_PD_8,\
LED_12_PD_1, LED_12_PD_2, LED_12_PD_3, LED_12_PD_4, LED_12_PD_5, LED_12_PD_6, LED_12_PD_7, LED_12_PD_8,\
LED_13_PD_1, LED_13_PD_2, LED_13_PD_3, LED_13_PD_4, LED_13_PD_5, LED_13_PD_6, LED_13_PD_7, LED_13_PD_8,\
LED_14_PD_1, LED_14_PD_2, LED_14_PD_3, LED_14_PD_4, LED_14_PD_5, LED_14_PD_6, LED_14_PD_7, LED_14_PD_8,\
LED_15_PD_1, LED_15_PD_2, LED_15_PD_3, LED_15_PD_4, LED_15_PD_5, LED_15_PD_6, LED_15_PD_7, LED_15_PD_8,\
LED_16_PD_1, LED_16_PD_2, LED_16_PD_3, LED_16_PD_4, LED_16_PD_5, LED_16_PD_6, LED_16_PD_7, LED_16_PD_8',fmt='%s', comments='')



while True:
    line = ser.readline()   # read a byte
    if line:
        string = line.decode().split()  # convert the byte string to  a unicode string
        if string[0].isnumeric():
            String=np.array([string])
            arr = String.astype('float64')
            B=np.append(B,arr,axis=0)
            n_LED_x_PD_y=B.reshape(-1,16,8)
            

            now = datetime.now()
            print(now)
            TimeStamp=np.array([now.timestamp(),0,0])
            
            rowvoltagedata=np.append(TimeStamp,arr)
            rowvoltagedata=rowvoltagedata.reshape(1,-1)
            with open(RowVoltage,'a') as csvfile:
                np.savetxt(csvfile, rowvoltagedata, delimiter=',',fmt='%f', comments='')
           
            if keyboard.is_pressed('esc'):
                break; 
                ser.close()

         
ser.close()
