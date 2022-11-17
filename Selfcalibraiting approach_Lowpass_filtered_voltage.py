# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:22:21 2022

@author: hmchayme
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter,filtfilt
from matplotlib import style
import random
import sys
import csv
import keyboard
import time
from IPython.display import clear_output
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from plotly.offline import plot
if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("Data"):
    os.mkdir("Data")
if not os.path.exists("filter"):
    os.mkdir("filter")    
# %% Filter definition
#Kalman Filter
def get_Kalman_gain(E_est_t1, E_mea):
    return (E_est_t1)/(E_est_t1+E_mea)

def get_Estimate(KG,est_t0,mea):
    return est_t0 + KG*(mea-est_t0)

def get_Error_of_estimate(KG, E_est_t0,Q):
    return (1-KG)*E_est_t0+Q
#Lowpass Filter
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

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


µs_forearm_690_780_810_850=np.array([µs_forearm_690,µs_forearm_780,µs_forearm_850])

# epsylon_lambda_Chro:  Extinction coefficient in (mM^-1* mm^-1) 
#Moaveni		and Landesman
citation='_HbR_HbO_Gratzer_6.5µM_ICG_Landesman'
epsylon_690_HbO=0.0276
epsylon_690_HbR=0.205196
epsylon_690_CtOx=0.263213
epsylon_690_ICG=2.8169 #6.5µM

epsylon_780_HbO=0.071
epsylon_780_HbR=0.107544
epsylon_780_CtOx=0.204948
epsylon_780_ICG=11.511 #6.55µM

epsylon_810_HbO=0.0864
epsylon_810_HbR=0.071708
epsylon_810_CtOx=0.231661
epsylon_810_ICG=19.072 #´6.5µM

epsylon_850_HbO=0.1058
epsylon_850_HbR=0.069132
epsylon_850_CtOx=0.22899
epsylon_850_ICG=2.9618 #6.5µM
#epsylon_lambda_chromophore=[   epsylon_690_HbO epsylon_690_HbR epsylon_690_CtOx  epsylon_690_ICG
#                               epsylon_780_HbO epsylon_780_HbR epsylon_780_CtOx  epsylon_780_ICG
#                               epsylon_810_HbO epsylon_810_HbR epsylon_810_CtOx  epsylon_810_ICG
#                               epsylon_850_HbO epsylon_850_HbR epsylon_850_CtOx  epsylon_850_ICG]


epsylon_690_780_810_850_HbO_HbR_CtOx_ICG=np.array([[epsylon_690_HbO, epsylon_690_HbR,  epsylon_690_ICG],
                                                          [epsylon_780_HbO, epsylon_780_HbR,  epsylon_780_ICG],
                                                          [epsylon_850_HbO, epsylon_850_HbR,  epsylon_850_ICG]])

B=[]
B1=[]
B2=[]

#E_mea=0.01
E_est_t1=0.005
E_est_t0=0.005
est_t0=2340
est_t1=2340

Q=0.0003
Error_factor=0.1
# %% Define and read the voltages    
headers = ['TStamp','Trg','Frame',
'LED_1_PD_1','LED_1_PD_2','LED_1_PD_3','LED_1_PD_4','LED_1_PD_5','LED_1_PD_6','LED_1_PD_7','LED_1_PD_8',
'LED_2_PD_1','LED_2_PD_2','LED_2_PD_3','LED_2_PD_4','LED_2_PD_5','LED_2_PD_6','LED_2_PD_7','LED_2_PD_8',
'LED_3_PD_1','LED_3_PD_2','LED_3_PD_3','LED_3_PD_4','LED_3_PD_5','LED_3_PD_6','LED_3_PD_7','LED_3_PD_8',
'LED_4_PD_1','LED_4_PD_2','LED_4_PD_3','LED_4_PD_4','LED_4_PD_5','LED_4_PD_6','LED_4_PD_7','LED_4_PD_8',
'LED_5_PD_1','LED_5_PD_2','LED_5_PD_3','LED_5_PD_4','LED_5_PD_5','LED_5_PD_6','LED_5_PD_7','LED_5_PD_8',
'LED_6_PD_1','LED_6_PD_2','LED_6_PD_3','LED_6_PD_4','LED_6_PD_5','LED_6_PD_6','LED_6_PD_7','LED_6_PD_8',
'LED_7_PD_1','LED_7_PD_2','LED_7_PD_3','LED_7_PD_4','LED_7_PD_5','LED_7_PD_6','LED_7_PD_7','LED_7_PD_8',
'LED_8_PD_1','LED_8_PD_2','LED_8_PD_3','LED_8_PD_4','LED_8_PD_5','LED_8_PD_6','LED_8_PD_7','LED_8_PD_8',
'LED_9_PD_1','LED_9_PD_2','LED_9_PD_3','LED_9_PD_4','LED_9_PD_5','LED_9_PD_6','LED_9_PD_7','LED_9_PD_8',
'LED_10_PD_1','LED_10_PD_2','LED_10_PD_3','LED_10_PD_4','LED_10_PD_5','LED_10_PD_6','LED_10_PD_7','LED_10_PD_8',
'LED_11_PD_1','LED_11_PD_2','LED_11_PD_3','LED_11_PD_4','LED_11_PD_5','LED_11_PD_6','LED_11_PD_7','LED_11_PD_8',
'LED_12_PD_1','LED_12_PD_2','LED_12_PD_3','LED_12_PD_4','LED_12_PD_5','LED_12_PD_6','LED_12_PD_7','LED_12_PD_8',
'LED_13_PD_1','LED_13_PD_2','LED_13_PD_3','LED_13_PD_4','LED_13_PD_5','LED_13_PD_6','LED_13_PD_7','LED_13_PD_8',
'LED_14_PD_1','LED_14_PD_2','LED_14_PD_3','LED_14_PD_4','LED_14_PD_5','LED_14_PD_6','LED_14_PD_7','LED_14_PD_8',
'LED_15_PD_1','LED_15_PD_2','LED_15_PD_3','LED_15_PD_4','LED_15_PD_5','LED_15_PD_6','LED_15_PD_7','LED_15_PD_8',
'LED_16_PD_1','LED_16_PD_2','LED_16_PD_3','LED_16_PD_4','LED_16_PD_5','LED_16_PD_6','LED_16_PD_7','LED_16_PD_8']
df = pd.read_csv('filter/Lowpass_filtered_Voltage_data'+citation+'.csv', dtype=str)
headers=np.array(headers)

headers=np.array(headers)
df_2=df[['TStamp',' Trg',' Frame']]

data=np.asarray(df)

# %% The voltages  in 2D matrix

data = data = data.astype('float')
data_voltages_raw=data[:,3:]
(M,N)=data_voltages_raw.shape
N=N

subcol=4
subrow=int(N/4)
subtitle=['LED_1_PD_1','LED_1_PD_2','LED_1_PD_3','LED_1_PD_4','LED_1_PD_5','LED_1_PD_6','LED_1_PD_7','LED_1_PD_8',
'LED_2_PD_1','LED_2_PD_2','LED_2_PD_3','LED_2_PD_4','LED_2_PD_5','LED_2_PD_6','LED_2_PD_7','LED_2_PD_8',
'LED_3_PD_1','LED_3_PD_2','LED_3_PD_3','LED_3_PD_4','LED_3_PD_5','LED_3_PD_6','LED_3_PD_7','LED_3_PD_8',
'LED_4_PD_1','LED_4_PD_2','LED_4_PD_3','LED_4_PD_4','LED_4_PD_5','LED_4_PD_6','LED_4_PD_7','LED_4_PD_8',
'LED_5_PD_1','LED_5_PD_2','LED_5_PD_3','LED_5_PD_4','LED_5_PD_5','LED_5_PD_6','LED_5_PD_7','LED_5_PD_8',
'LED_6_PD_1','LED_6_PD_2','LED_6_PD_3','LED_6_PD_4','LED_6_PD_5','LED_6_PD_6','LED_6_PD_7','LED_6_PD_8',
'LED_7_PD_1','LED_7_PD_2','LED_7_PD_3','LED_7_PD_4','LED_7_PD_5','LED_7_PD_6','LED_7_PD_7','LED_7_PD_8',
'LED_8_PD_1','LED_8_PD_2','LED_8_PD_3','LED_8_PD_4','LED_8_PD_5','LED_8_PD_6','LED_8_PD_7','LED_8_PD_8',
'LED_9_PD_1','LED_9_PD_2','LED_9_PD_3','LED_9_PD_4','LED_9_PD_5','LED_9_PD_6','LED_9_PD_7','LED_9_PD_8',
'LED_10_PD_1','LED_10_PD_2','LED_10_PD_3','LED_10_PD_4','LED_10_PD_5','LED_10_PD_6','LED_10_PD_7','LED_10_PD_8',
'LED_11_PD_1','LED_11_PD_2','LED_11_PD_3','LED_11_PD_4','LED_11_PD_5','LED_11_PD_6','LED_11_PD_7','LED_11_PD_8',
'LED_12_PD_1','LED_12_PD_2','LED_12_PD_3','LED_12_PD_4','LED_12_PD_5','LED_12_PD_6','LED_12_PD_7','LED_12_PD_8',
'LED_13_PD_1','LED_13_PD_2','LED_13_PD_3','LED_13_PD_4','LED_13_PD_5','LED_13_PD_6','LED_13_PD_7','LED_13_PD_8',
'LED_14_PD_1','LED_14_PD_2','LED_14_PD_3','LED_14_PD_4','LED_14_PD_5','LED_14_PD_6','LED_14_PD_7','LED_14_PD_8',
'LED_15_PD_1','LED_15_PD_2','LED_15_PD_3','LED_15_PD_4','LED_15_PD_5','LED_15_PD_6','LED_15_PD_7','LED_15_PD_8',
'LED_16_PD_1','LED_16_PD_2','LED_16_PD_3','LED_16_PD_4','LED_16_PD_5','LED_16_PD_6','LED_16_PD_7','LED_16_PD_8']
#%% The voltages  in 3D matrix
data_voltages_raw=data_voltages_raw
n_LED_x_PD_y=data_voltages_raw.reshape(-1,16,8)
# %% Calculation oh the slops 
    
Sl_PD_1_4_690_R=((0.5*(np.log((n_LED_x_PD_y[:,4,0]*n_LED_x_PD_y[:,3,3])/(n_LED_x_PD_y[:,4,3]*n_LED_x_PD_y[:,3,0]))))+(2*np.log(32.14/8.54)))/(32.14-8.54)
Sl_PD_1_4_780_R=((0.5*(np.log((n_LED_x_PD_y[:,5,0]*n_LED_x_PD_y[:,2,3])/(n_LED_x_PD_y[:,5,3]*n_LED_x_PD_y[:,2,0]))))+(2*np.log(33.14/9.49)))/(33.14-9.49)
Sl_PD_1_4_810_R=((0.5*(np.log((n_LED_x_PD_y[:,6,0]*n_LED_x_PD_y[:,1,3])/(n_LED_x_PD_y[:,6,3]*n_LED_x_PD_y[:,1,0]))))+(2*np.log(34.13/10.33)))/(34.13-10.33)
Sl_PD_1_4_850_R=((0.5*(np.log((n_LED_x_PD_y[:,7,0]*n_LED_x_PD_y[:,0,3])/(n_LED_x_PD_y[:,7,3]*n_LED_x_PD_y[:,0,0]))))+(2*np.log(35.13/11.40)))/(35.13-11.40)

Sl_PD_2_3_690_R=((0.5*(np.log((n_LED_x_PD_y[:,4,1]*n_LED_x_PD_y[:,3,2])/(n_LED_x_PD_y[:,4,2]*n_LED_x_PD_y[:,3,1]))))+(2*np.log(24.19/16.28)))/(24.19-16.28)
Sl_PD_2_3_780_R=((0.5*(np.log((n_LED_x_PD_y[:,5,1]*n_LED_x_PD_y[:,2,2])/(n_LED_x_PD_y[:,5,2]*n_LED_x_PD_y[:,2,1]))))+(2*np.log(25.18/17.26)))/(25.18-17.26)
Sl_PD_2_3_810_R=((0.5*(np.log((n_LED_x_PD_y[:,6,1]*n_LED_x_PD_y[:,1,2])/(n_LED_x_PD_y[:,6,2]*n_LED_x_PD_y[:,1,1]))))+(2*np.log(26.17/18.25)))/(26.17-18.25)
Sl_PD_2_3_850_R=((0.5*(np.log((n_LED_x_PD_y[:,7,1]*n_LED_x_PD_y[:,0,2])/(n_LED_x_PD_y[:,7,2]*n_LED_x_PD_y[:,0,1]))))+(2*np.log(27.17/19.24)))/(27.17-19.24)

Sl_PD_1_4_690_L=((0.5*(np.log((n_LED_x_PD_y[:,12,3]*n_LED_x_PD_y[:,11,0])/(n_LED_x_PD_y[:,12,0]*n_LED_x_PD_y[:,11,3]))))+(2*np.log(32.56/10)))/(32.56-10)
Sl_PD_1_4_780_L=((0.5*(np.log((n_LED_x_PD_y[:,13,3]*n_LED_x_PD_y[:,10,0])/(n_LED_x_PD_y[:,13,0]*n_LED_x_PD_y[:,10,3]))))+(2*np.log(33.54/10.82)))/(33.54-10.82)
Sl_PD_1_4_810_L=((0.5*(np.log((n_LED_x_PD_y[:,14,3]*n_LED_x_PD_y[:,9,0])/(n_LED_x_PD_y[:,14,0]*n_LED_x_PD_y[:,9,3]))))+(2*np.log(34.53/11.66)))/(34.53-11.66)
Sl_PD_1_4_850_L=((0.5*(np.log((n_LED_x_PD_y[:,15,3]*n_LED_x_PD_y[:,8,0])/(n_LED_x_PD_y[:,15,0]*n_LED_x_PD_y[:,8,3]))))+(2*np.log(35.51/12.53)))/(35.51-12.53)

Sl_PD_2_3_690_L=((0.5*(np.log((n_LED_x_PD_y[:,12,2]*n_LED_x_PD_y[:,11,1])/(n_LED_x_PD_y[:,12,1]*n_LED_x_PD_y[:,11,2]))))+(2*np.log(24.74/17.09)))/(24.74-17.09)
Sl_PD_2_3_780_L=((0.5*(np.log((n_LED_x_PD_y[:,13,2]*n_LED_x_PD_y[:,10,1])/(n_LED_x_PD_y[:,13,1]*n_LED_x_PD_y[:,10,2]))))+(2*np.log(25.71/18.03)))/(25.71-18.03)
Sl_PD_2_3_810_L=((0.5*(np.log((n_LED_x_PD_y[:,14,2]*n_LED_x_PD_y[:,9,1])/(n_LED_x_PD_y[:,14,1]*n_LED_x_PD_y[:,9,2]))))+(2*np.log(26.68/18.97)))/(26.68-18.97)
Sl_PD_2_3_850_L=((0.5*(np.log((n_LED_x_PD_y[:,15,2]*n_LED_x_PD_y[:,8,1])/(n_LED_x_PD_y[:,15,1]*n_LED_x_PD_y[:,8,2]))))+(2*np.log(27.66/19.92)))/(27.66-19.92)

Sl_PD_5_8_690_R=((0.5*(np.log((n_LED_x_PD_y[:,4,7]*n_LED_x_PD_y[:,3,4])/(n_LED_x_PD_y[:,4,4]*n_LED_x_PD_y[:,3,7]))))+(2*np.log(32.56/10)))/(32.56-10)
Sl_PD_5_8_780_R=((0.5*(np.log((n_LED_x_PD_y[:,5,7]*n_LED_x_PD_y[:,2,4])/(n_LED_x_PD_y[:,5,4]*n_LED_x_PD_y[:,2,7]))))+(2*np.log(33.54/10.82)))/(33.54-10.82)
Sl_PD_5_8_810_R=((0.5*(np.log((n_LED_x_PD_y[:,6,7]*n_LED_x_PD_y[:,1,4])/(n_LED_x_PD_y[:,6,4]*n_LED_x_PD_y[:,1,7]))))+(2*np.log(34.53/11.66)))/(34.53-11.66)
Sl_PD_5_8_850_R=((0.5*(np.log((n_LED_x_PD_y[:,7,7]*n_LED_x_PD_y[:,0,4])/(n_LED_x_PD_y[:,7,4]*n_LED_x_PD_y[:,0,7]))))+(2*np.log(35.51/12.53)))/(35.51-12.53)

Sl_PD_6_7_690_R=((0.5*(np.log((n_LED_x_PD_y[:,4,6]*n_LED_x_PD_y[:,3,5])/(n_LED_x_PD_y[:,4,5]*n_LED_x_PD_y[:,3,6]))))+(2*np.log(24.74/17.09)))/(24.74-17.09)
Sl_PD_6_7_780_R=((0.5*(np.log((n_LED_x_PD_y[:,5,6]*n_LED_x_PD_y[:,2,5])/(n_LED_x_PD_y[:,5,5]*n_LED_x_PD_y[:,2,6]))))+(2*np.log(25.71/18.03)))/(25.71-18.03)
Sl_PD_6_7_810_R=((0.5*(np.log((n_LED_x_PD_y[:,6,6]*n_LED_x_PD_y[:,1,5])/(n_LED_x_PD_y[:,6,5]*n_LED_x_PD_y[:,1,6]))))+(2*np.log(26.68/18.97)))/(26.68-18.97)
Sl_PD_6_7_850_R=((0.5*(np.log((n_LED_x_PD_y[:,7,6]*n_LED_x_PD_y[:,0,5])/(n_LED_x_PD_y[:,7,5]*n_LED_x_PD_y[:,0,6]))))+(2*np.log(27.66/19.92)))/(27.66-19.92)
            
Sl_PD_5_8_690_L=((0.5*(np.log((n_LED_x_PD_y[:,12,4]*n_LED_x_PD_y[:,11,7])/(n_LED_x_PD_y[:,12,7]*n_LED_x_PD_y[:,11,4]))))+(2*np.log(32.14/8.54)))/(32.14-8.54)
Sl_PD_5_8_780_L=((0.5*(np.log((n_LED_x_PD_y[:,13,4]*n_LED_x_PD_y[:,10,7])/(n_LED_x_PD_y[:,13,7]*n_LED_x_PD_y[:,10,4]))))+(2*np.log(33.14/9.49)))/(33.14-9.49)
Sl_PD_5_8_810_L=((0.5*(np.log((n_LED_x_PD_y[:,14,4]*n_LED_x_PD_y[:,9,7])/(n_LED_x_PD_y[:,14,7]*n_LED_x_PD_y[:,9,4]))))+(2*np.log(34.13/10.33)))/(34.13-10.33)
Sl_PD_5_8_850_L=((0.5*(np.log((n_LED_x_PD_y[:,15,4]*n_LED_x_PD_y[:,8,7])/(n_LED_x_PD_y[:,15,7]*n_LED_x_PD_y[:,8,4]))))+(2*np.log(35.13/11.40)))/(35.13-11.40)

Sl_PD_6_7_690_L=((0.5*(np.log((n_LED_x_PD_y[:,12,5]*n_LED_x_PD_y[:,11,6])/(n_LED_x_PD_y[:,12,6]*n_LED_x_PD_y[:,11,5]))))+(2*np.log(24.19/16.28)))/(24.19-16.28)
Sl_PD_6_7_780_L=((0.5*(np.log((n_LED_x_PD_y[:,13,5]*n_LED_x_PD_y[:,10,6])/(n_LED_x_PD_y[:,13,6]*n_LED_x_PD_y[:,10,5]))))+(2*np.log(25.18/17.26)))/(25.18-17.26)
Sl_PD_6_7_810_L=((0.5*(np.log((n_LED_x_PD_y[:,14,5]*n_LED_x_PD_y[:,9,6])/(n_LED_x_PD_y[:,14,6]*n_LED_x_PD_y[:,9,5]))))+(2*np.log(26.17/18.25)))/(26.17-18.25)
Sl_PD_6_7_850_L=((0.5*(np.log((n_LED_x_PD_y[:,15,5]*n_LED_x_PD_y[:,8,6])/(n_LED_x_PD_y[:,15,6]*n_LED_x_PD_y[:,8,5]))))+(2*np.log(27.17/19.24)))/(27.17-19.24)





Sl_PD_1_8_690_U=((0.5*(np.log((n_LED_x_PD_y[:,3,7]*n_LED_x_PD_y[:,12,0])/(n_LED_x_PD_y[:,3,0]*n_LED_x_PD_y[:,12,7]))))+(2*np.log(10/8.54)))/(10-8.54)
Sl_PD_1_8_780_U=((0.5*(np.log((n_LED_x_PD_y[:,2,7]*n_LED_x_PD_y[:,13,0])/(n_LED_x_PD_y[:,2,0]*n_LED_x_PD_y[:,13,7]))))+(2*np.log(10.82/9.49)))/(10.82-9.49)
Sl_PD_1_8_810_U=((0.5*(np.log((n_LED_x_PD_y[:,1,7]*n_LED_x_PD_y[:,14,0])/(n_LED_x_PD_y[:,1,0]*n_LED_x_PD_y[:,14,7]))))+(2*np.log(11.66/10.44)))/(11.66-10.44)
Sl_PD_1_8_850_U=((0.5*(np.log((n_LED_x_PD_y[:,0,7]*n_LED_x_PD_y[:,15,0])/(n_LED_x_PD_y[:,0,0]*n_LED_x_PD_y[:,15,7]))))+(2*np.log(12.53/11.40)))/(12.53-11.40)

Sl_PD_2_7_690_U=((0.5*(np.log((n_LED_x_PD_y[:,3,6]*n_LED_x_PD_y[:,12,1])/(n_LED_x_PD_y[:,3,1]*n_LED_x_PD_y[:,12,6]))))+(2*np.log(17.09/16.28)))/(17.09-16.28)
Sl_PD_2_7_780_U=((0.5*(np.log((n_LED_x_PD_y[:,2,6]*n_LED_x_PD_y[:,13,1])/(n_LED_x_PD_y[:,2,1]*n_LED_x_PD_y[:,13,6]))))+(2*np.log(18.03/17.26)))/(18.03-17.26)
Sl_PD_2_7_810_U=((0.5*(np.log((n_LED_x_PD_y[:,1,6]*n_LED_x_PD_y[:,14,1])/(n_LED_x_PD_y[:,1,1]*n_LED_x_PD_y[:,14,6]))))+(2*np.log(18.97/18.25)))/(18.97-18.25)
Sl_PD_2_7_850_U=((0.5*(np.log((n_LED_x_PD_y[:,0,6]*n_LED_x_PD_y[:,15,1])/(n_LED_x_PD_y[:,0,1]*n_LED_x_PD_y[:,15,6]))))+(2*np.log(19.92/19.24)))/(19.92-19.24)

Sl_PD_3_6_690_U=((0.5*(np.log((n_LED_x_PD_y[:,3,5]*n_LED_x_PD_y[:,12,2])/(n_LED_x_PD_y[:,3,2]*n_LED_x_PD_y[:,12,5]))))+(2*np.log(24.74/24.19)))/(24.74-24.19)
Sl_PD_3_6_780_U=((0.5*(np.log((n_LED_x_PD_y[:,2,5]*n_LED_x_PD_y[:,13,2])/(n_LED_x_PD_y[:,2,2]*n_LED_x_PD_y[:,13,5]))))+(2*np.log(25.71/25.18)))/(25.71-25.18)
Sl_PD_3_6_810_U=((0.5*(np.log((n_LED_x_PD_y[:,1,5]*n_LED_x_PD_y[:,14,2])/(n_LED_x_PD_y[:,1,2]*n_LED_x_PD_y[:,14,5]))))+(2*np.log(26.68/26.17)))/(26.68-26.17)
Sl_PD_3_6_850_U=((0.5*(np.log((n_LED_x_PD_y[:,0,5]*n_LED_x_PD_y[:,15,2])/(n_LED_x_PD_y[:,0,2]*n_LED_x_PD_y[:,15,5]))))+(2*np.log(27.66/27.17)))/(27.66-27.17)

Sl_PD_4_5_690_U=((0.5*(np.log((n_LED_x_PD_y[:,3,4]*n_LED_x_PD_y[:,12,3])/(n_LED_x_PD_y[:,3,3]*n_LED_x_PD_y[:,12,4]))))+(2*np.log(32.56/32.14)))/(32.56-32.14)
Sl_PD_4_5_780_U=((0.5*(np.log((n_LED_x_PD_y[:,2,4]*n_LED_x_PD_y[:,13,3])/(n_LED_x_PD_y[:,2,3]*n_LED_x_PD_y[:,13,4]))))+(2*np.log(33.54/33.14)))/(33.54-33.14)
Sl_PD_4_5_810_U=((0.5*(np.log((n_LED_x_PD_y[:,1,4]*n_LED_x_PD_y[:,14,3])/(n_LED_x_PD_y[:,1,3]*n_LED_x_PD_y[:,14,4]))))+(2*np.log(34.53/34.13)))/(34.53-34.13)
Sl_PD_4_5_850_U=((0.5*(np.log((n_LED_x_PD_y[:,0,4]*n_LED_x_PD_y[:,15,3])/(n_LED_x_PD_y[:,0,3]*n_LED_x_PD_y[:,15,4]))))+(2*np.log(35.51/35.13)))/(35.51-35.13)

Sl_PD_1_8_690_D=((0.5*(np.log((n_LED_x_PD_y[:,4,7]*n_LED_x_PD_y[:,11,0])/(n_LED_x_PD_y[:,4,0]*n_LED_x_PD_y[:,11,7]))))+(2*np.log(32.56/32.14)))/(32.56-32.14)
Sl_PD_1_8_780_D=((0.5*(np.log((n_LED_x_PD_y[:,5,7]*n_LED_x_PD_y[:,10,0])/(n_LED_x_PD_y[:,5,0]*n_LED_x_PD_y[:,10,7]))))+(2*np.log(33.54/33.14)))/(33.54-33.14)
Sl_PD_1_8_810_D=((0.5*(np.log((n_LED_x_PD_y[:,6,7]*n_LED_x_PD_y[:,9,0])/(n_LED_x_PD_y[:,6,0]*n_LED_x_PD_y[:,9,7]))))+(2*np.log(34.53/34.13)))/(34.53-34.13)
Sl_PD_1_8_850_D=((0.5*(np.log((n_LED_x_PD_y[:,7,7]*n_LED_x_PD_y[:,8,0])/(n_LED_x_PD_y[:,7,0]*n_LED_x_PD_y[:,8,7]))))+(2*np.log(35.51/35.13)))/(35.51-35.13)

Sl_PD_2_7_690_D=((0.5*(np.log((n_LED_x_PD_y[:,4,6]*n_LED_x_PD_y[:,11,1])/(n_LED_x_PD_y[:,4,1]*n_LED_x_PD_y[:,11,6]))))+(2*np.log(24.74/24.19)))/(24.74-24.19)
Sl_PD_2_7_780_D=((0.5*(np.log((n_LED_x_PD_y[:,5,6]*n_LED_x_PD_y[:,10,1])/(n_LED_x_PD_y[:,5,1]*n_LED_x_PD_y[:,10,6]))))+(2*np.log(25.71/25.18)))/(25.71-25.18)
Sl_PD_2_7_810_D=((0.5*(np.log((n_LED_x_PD_y[:,6,6]*n_LED_x_PD_y[:,9,1])/(n_LED_x_PD_y[:,6,1]*n_LED_x_PD_y[:,9,6]))))+(2*np.log(26.68/26.17)))/(26.68-26.17)
Sl_PD_2_7_850_D=((0.5*(np.log((n_LED_x_PD_y[:,7,6]*n_LED_x_PD_y[:,8,1])/(n_LED_x_PD_y[:,7,1]*n_LED_x_PD_y[:,8,6]))))+(2*np.log(27.66/27.17)))/(27.66-27.17)

Sl_PD_3_6_690_D=((0.5*(np.log((n_LED_x_PD_y[:,4,5]*n_LED_x_PD_y[:,11,2])/(n_LED_x_PD_y[:,4,2]*n_LED_x_PD_y[:,11,5]))))+(2*np.log(17.09/16.28)))/(17.09-16.28)
Sl_PD_3_6_780_D=((0.5*(np.log((n_LED_x_PD_y[:,5,5]*n_LED_x_PD_y[:,10,2])/(n_LED_x_PD_y[:,5,2]*n_LED_x_PD_y[:,10,5]))))+(2*np.log(18.03/17.26)))/(18.03-17.26)
Sl_PD_3_6_810_D=((0.5*(np.log((n_LED_x_PD_y[:,6,5]*n_LED_x_PD_y[:,9,2])/(n_LED_x_PD_y[:,6,2]*n_LED_x_PD_y[:,9,5]))))+(2*np.log(18.97/18.25)))/(18.97-18.25)
Sl_PD_3_6_850_D=((0.5*(np.log((n_LED_x_PD_y[:,7,5]*n_LED_x_PD_y[:,8,2])/(n_LED_x_PD_y[:,7,2]*n_LED_x_PD_y[:,8,5]))))+(2*np.log(19.92/19.24)))/(19.92-19.24)

Sl_PD_4_5_690_D=((0.5*(np.log((n_LED_x_PD_y[:,4,4]*n_LED_x_PD_y[:,11,3])/(n_LED_x_PD_y[:,4,3]*n_LED_x_PD_y[:,11,4]))))+(2*np.log(10/8.54)))/(10-8.54)
Sl_PD_4_5_780_D=((0.5*(np.log((n_LED_x_PD_y[:,5,4]*n_LED_x_PD_y[:,10,3])/(n_LED_x_PD_y[:,5,3]*n_LED_x_PD_y[:,10,4]))))+(2*np.log(10.82/9.49)))/(10.82-9.49)
Sl_PD_4_5_810_D=((0.5*(np.log((n_LED_x_PD_y[:,6,4]*n_LED_x_PD_y[:,9,3])/(n_LED_x_PD_y[:,6,3]*n_LED_x_PD_y[:,9,4]))))+(2*np.log(11.66/10.44)))/(11.66-10.44)
Sl_PD_4_5_850_D=((0.5*(np.log((n_LED_x_PD_y[:,7,4]*n_LED_x_PD_y[:,8,3])/(n_LED_x_PD_y[:,7,3]*n_LED_x_PD_y[:,8,4]))))+(2*np.log(12.53/11.40)))/(12.53-11.40)
# %% Slops matrix
Sl=np.array([
Sl_PD_1_4_690_R,Sl_PD_1_4_780_R,Sl_PD_1_4_850_R,
Sl_PD_2_3_690_R,Sl_PD_2_3_780_R,Sl_PD_2_3_850_R,
Sl_PD_1_4_690_L,Sl_PD_1_4_780_L,Sl_PD_1_4_850_L,
Sl_PD_2_3_690_L,Sl_PD_2_3_780_L,Sl_PD_2_3_850_L,
Sl_PD_5_8_690_R,Sl_PD_5_8_780_R,Sl_PD_5_8_850_R,
Sl_PD_6_7_690_R,Sl_PD_6_7_780_R,Sl_PD_6_7_850_R,
Sl_PD_5_8_690_L,Sl_PD_5_8_780_L,Sl_PD_5_8_850_L,
Sl_PD_6_7_690_L,Sl_PD_6_7_780_L,Sl_PD_6_7_850_L,
Sl_PD_1_8_690_U,Sl_PD_1_8_780_U,Sl_PD_1_8_850_U,
Sl_PD_2_7_690_U,Sl_PD_2_7_780_U,Sl_PD_2_7_850_U,
Sl_PD_3_6_690_U,Sl_PD_3_6_780_U,Sl_PD_3_6_850_U,
Sl_PD_4_5_690_U,Sl_PD_4_5_780_U,Sl_PD_4_5_850_U,
Sl_PD_1_8_690_D,Sl_PD_1_8_780_D,Sl_PD_1_8_850_D,
Sl_PD_2_7_690_D,Sl_PD_2_7_780_D,Sl_PD_2_7_850_D,
Sl_PD_3_6_690_D,Sl_PD_3_6_780_D,Sl_PD_3_6_850_D,
Sl_PD_4_5_690_D,Sl_PD_4_5_780_D,Sl_PD_4_5_850_D]).T


# %% Caculation of the raw Concentration data
Sl_690_780_810_850= Sl.reshape(-1,16,3)

Sl_µs_forearm_690_780_810_850=(Sl_690_780_810_850**2)/(3*µs_forearm_690_780_810_850)  
# µ_a   
Sl_µs_forearm_690_780_810_850_re= np.transpose(Sl_µs_forearm_690_780_810_850, (0,2,1))
Abs_Concentration_690_780_810_850_HbO_HbR_CtOx_ICG= (1/np.log(10))*np.matmul(np.linalg.inv(epsylon_690_780_810_850_HbO_HbR_CtOx_ICG),Sl_µs_forearm_690_780_810_850_re)
Abs_Concentration_690_780_810_850_HbO_HbR_CtOx_ICG=np.transpose(Abs_Concentration_690_780_810_850_HbO_HbR_CtOx_ICG, (0,2,1))


Concentrationrawvoltagedata ="filter/Concentration_Lowpass_filtered_voltage_data "+citation+".csv"#+ time.strftime("%Y%m%d-%H%M%S")
Concentration_raw_voltage_data=Abs_Concentration_690_780_810_850_HbO_HbR_CtOx_ICG.reshape(-1,48)
'''
with open(Concentrationrawvoltagedata,'a') as csvfile:
                np.savetxt(csvfile, Concentration_raw_voltage_data,header='\
HbO_S01D01D04S02,HbR_S01D01D04S02,CtOx_S01D01D04S02,ICG_S01D01D04S02,\
HbO_S01D02D03S02,HbR_S01D02D03S02,CtOx_S01D02D03S02,ICG_S01D02D03S02,\
HbO_S04D01D04S03,HbR_S04D01D04S03,CtOx_S04D01D04S03,ICG_S04D01D04S03,\
HbO_S04D02D03S03,HbR_S04D02D03S03,CtOx_S04D02D03S03,ICG_S04D02D03S03,\
HbO_S01D05D08S02,HbR_S01D05D08S02,CtOx_S01D05D08S02,ICG_S01D05D08S02,\
HbO_S01D06D07S02,HbR_S01D06D07S02,CtOx_S01D06D07S02,ICG_S01D06D07S02,\
HbO_S04D05D08S03,HbR_S04D05D08S03,CtOx_S04D05D08S03,ICG_S04D05D08S03,\
HbO_S04D06D07S03,HbR_S04D06D07S03,CtOx_S04D06D07S03,ICG_S04D06D07S03,\
HbO_S01D0108S04,HbR_S01D0108S04,CtOx_S01D0108S04,ICG_S01D0108S04,\
HbO_S01D0207S04,HbR_S01D0207S04,CtOx_S01D0207S04,ICG_S01D0207S04,\
HbO_S01D0306S04,HbR_S01D0306S04,CtOx_S01D0306S04,ICG_S01D0306S04,\
HbO_S01D0405S04,HbR_S01D0405S04,CtOx_S01D0405S04,ICG_S01D0405S04,\
HbO_S02D0108S03,HbR_S02D0108S03,CtOx_S02D0108S03,ICG_S02D0108S03,\
HbO_S02D0207S03,HbR_S02D0207S03,CtOx_S02D0207S03,ICG_S02D0207S03,\
HbO_S02D0306S03,HbR_S02D0306S03,CtOx_S02D0306S03,ICG_S02D0306S03,\
HbO_S02D0405S03,HbR_S02D0405S03,CtOx_S02D0405S03,ICG_S02D0405S03', delimiter=',',fmt='%f', comments='')
'''

dfobj = pd.DataFrame(Concentration_raw_voltage_data,columns = ['\
HbO_S01D01D04S02','HbR_S01D01D04S02','ICG_S01D01D04S02','\
HbO_S01D02D03S02','HbR_S01D02D03S02','ICG_S01D02D03S02','\
HbO_S04D01D04S03','HbR_S04D01D04S03','ICG_S04D01D04S03','\
HbO_S04D02D03S03','HbR_S04D02D03S03','ICG_S04D02D03S03','\
HbO_S01D05D08S02','HbR_S01D05D08S02','ICG_S01D05D08S02','\
HbO_S01D06D07S02','HbR_S01D06D07S02','ICG_S01D06D07S02','\
HbO_S04D05D08S03','HbR_S04D05D08S03','ICG_S04D05D08S03','\
HbO_S04D06D07S03','HbR_S04D06D07S03','ICG_S04D06D07S03','\
HbO_S01D01D08S04','HbR_S01D01D08S04','ICG_S01D01D08S04','\
HbO_S01D02D07S04','HbR_S01D02D07S04','ICG_S01D02D07S04','\
HbO_S01D03D06S04','HbR_S01D03D06S04','ICG_S01D03D06S04','\
HbO_S01D04D05S04','HbR_S01D04D05S04','ICG_S01D04D05S04','\
HbO_S02D01D08S03','HbR_S02D01D08S03','ICG_S02D01D08S03','\
HbO_S02D02D07S03','HbR_S02D02D07S03','ICG_S02D02D07S03','\
HbO_S02D03D06S03','HbR_S02D03D06S03','ICG_S02D03D06S03','\
HbO_S02D04D05S03','HbR_S02D04D05S03','ICG_S02D04D05S03'])

Concentration_raw_voltage_data = pd.concat([df_2, dfobj], axis=1)
Concentration_raw_voltage_data.to_csv(Concentrationrawvoltagedata,index=False)  

# %% Raw Concentration data in 2D matrix
'''Abs_Concentration_690_780_810_850_HbO_HbR_CtOx_ICG=Abs_Concentration_690_780_810_850_HbO_HbR_CtOx_ICG.reshape(-1,64)
#%%

fig = make_subplots(
    rows=subrow, cols=subcol,
    subplot_titles=subtitle)
subtitle=np.reshape(subtitle,([subrow,subcol]))


xs=np.arange(0,M)

subplot_row_col=np.arange(1,N+1).reshape([int(N/4),-1])
(subplot_row,subplot_col)=subplot_row_col.shape
data_reshaped=data_voltages_raw.reshape([-1,int(N/4),4])
data_set=data_reshaped[:,1,1]
kalman_filtered_signal=np.zeros(len(xs))

kalman_filtered_Voltage_data=np.zeros((M,1))
Lowpass_filtered_Voltage_data=np.zeros((M,1))
kalman_filtered_Voltage_data=np.empty((M,1))
Lowpass_filtered_Voltage_data=np.empty((M,1))
            
#%%the ploted filterd voltage data

for i in range(0, subplot_row):
    for j in range(0, subplot_col):
        ys=data_reshaped[:,i,j]
        
        T = len(ys)/3         # Sample Period
        fs = 3       # sample rate, Hz
        cutoff = 0.02      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 5       # sin wave can be approx represented as quadratic
        n = int(T * fs) # total number of samples
        Lowpass_filtered_signal = butter_lowpass_filter(ys, cutoff, fs, order)
        
        
        est_t0=np.mean(ys[0:20])
        est_t1=np.mean(ys[0:20])


        for loop in range(len(ys)):
        
            mea=ys[loop]
            
            E_mea = (np.abs(ys[loop]-est_t0)*Error_factor)+Q
            #E_mea = (np.abs(distance[loop]-est_t0)*10**-3)+(0.001)
            KG = get_Kalman_gain(E_est_t1, E_mea)
            
            est_t1 = get_Estimate(KG, est_t0, mea)
            E_est_t1 = get_Error_of_estimate(KG, E_est_t0,Q)
            E_est_t0 = E_est_t1
            est_t0 = est_t1
            
            kalman_filtered_signal[loop]=est_t1
            
        kalman_filtered_Voltage_data=np.append(kalman_filtered_Voltage_data,kalman_filtered_signal.reshape(M,-1),axis=1)
        Lowpass_filtered_Voltage_data=np.append(Lowpass_filtered_Voltage_data,Lowpass_filtered_signal.reshape(M,-1),axis=1)
        
        fig.add_trace(go.Scatter(x=xs,y=ys,
                     mode='lines',line_color="crimson",
                     line=dict(width=1),
                     name=subtitle[i,j]),                    
           row=i+1,col=j+1)
        
        fig.add_trace(go.Scatter(x=xs,y=kalman_filtered_signal,
                     mode='lines',line_color="gray",
                     line=dict(width=1),
                     name=subtitle[i,j]+'_Kalman_filter'),
           row=i+1,col=j+1)
        
        fig.add_trace(go.Scatter(x=xs,y = Lowpass_filtered_signal,
                     mode='lines',line_color="black",
                     line =dict(width=1),
                     name=subtitle[i,j]+'_Lowpass_filter'),
           row=i+1,col=j+1)
        fig.update_xaxes(title_text="Time in s", row=i+1, col=j+1)
        fig.update_yaxes(title_text="Voltage", row=i+1, col=j+1)  
fig.update_layout(width=1700,height=5000,title_text="Enable Pig 05",plot_bgcolor="rgb(256,256,256)")



                

plot(fig,filename="Lowpass_filtered_voltage.html")
fig.show()
fig.write_image("images/Lowpass_filtered_voltage.svg")
fig.write_html("html/Lowpass_filtered_voltage.html")'''