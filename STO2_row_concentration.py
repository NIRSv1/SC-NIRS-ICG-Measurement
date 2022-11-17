# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:24:37 2022

@author: hmchayme
"""

import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
import csv
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
def get_Kalman_gain(E_est_t1, E_mea):
    return (E_est_t1)/(E_est_t1+E_mea)

def get_Estimate(KG,est_t0,mea):
    return est_t0 + KG*(mea-est_t0)

def get_Error_of_estimate(KG, E_est_t0,Q):
    return (1-KG)*E_est_t0+Q

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
#E_mea=0.01
E_est_t1=0.005
E_est_t0=0.005
est_t0=2340
est_t1=2340

Q=0.00003
Error_factor=2
    
headers = ['Concentration_HbO_PD_1_4_LED_R','Concentration_HbR_PD_1_4_LED_R','Concentration_CtOx_PD_1_4_LED_R','Concentration_ICG_PD_1_4_LED_R',
           'Concentration_HbO_PD_2_3_LED_R','Concentration_HbR_PD_2_3_LED_R','Concentration_CtOx_PD_2_3_LED_R','Concentration_ICG_PD_2_3_LED_R',
           'Concentration_HbO_PD_1_4_LED_L','Concentration_HbR_PD_1_4_LED_L','Concentration_CtOx_PD_1_4_LED_L','Concentration_ICG_PD_1_4_LED_L',
           'Concentration_HbO_PD_2_3_LED_L','Concentration_HbR_PD_2_3_LED_L','Concentration_CtOx_PD_2_3_LED_L','Concentration_ICG_PD_2_3_LED_L',
           'Concentration_HbO_PD_5_8_LED_R','Concentration_HbR_PD_5_8_LED_R','Concentration_CtOx_PD_5_8_LED_R','Concentration_ICG_PD_5_8_LED_R',
           'Concentration_HbO_PD_6_7_LED_R','Concentration_HbR_PD_6_7_LED_R','Concentration_CtOx_PD_6_7_LED_R','Concentration_ICG_PD_6_7_LED_R',
           'Concentration_HbO_PD_5_8_LED_L','Concentration_HbR_PD_5_8_LED_L','Concentration_CtOx_PD_5_8_LED_L','Concentration_ICG_PD_5_8_LED_L',
           'Concentration_HbO_PD_6_7_LED_L','Concentration_HbR_PD_6_7_LED_L','Concentration_CtOx_PD_6_7_LED_L','Concentration_ICG_PD_6_7_LED_L',
           'Concentration_HbO_PD_1_8_LED_U','Concentration_HbR_PD_1_8_LED_U','Concentration_CtOx_PD_1_8_LED_U','Concentration_ICG_PD_1_8_LED_U',
           'Concentration_HbO_PD_2_7_LED_U','Concentration_HbR_PD_2_7_LED_U','Concentration_CtOx_PD_2_7_LED_U','Concentration_ICG_PD_2_7_LED_U',
           'Concentration_HbO_PD_3_6_LED_U','Concentration_HbR_PD_3_6_LED_U','Concentration_CtOx_PD_3_6_LED_U','Concentration_ICG_PD_3_6_LED_U',
           'Concentration_HbO_PD_4_5_LED_U','Concentration_HbR_PD_4_5_LED_U','Concentration_CtOx_PD_4_5_LED_U','Concentration_ICG_PD_4_5_LED_U',
           'Concentration_HbO_PD_1_8_LED_D','Concentration_HbR_PD_1_8_LED_D','Concentration_CtOx_PD_1_8_LED_D','Concentration_ICG_PD_1_8_LED_D',
           'Concentration_HbO_PD_2_7_LED_D','Concentration_HbR_PD_2_7_LED_D','Concentration_CtOx_PD_2_7_LED_D','Concentration_ICG_PD_2_7_LED_D',
           'Concentration_HbO_PD_3_6_LED_D','Concentration_HbR_PD_3_6_LED_D','Concentration_CtOx_PD_3_6_LED_D','Concentration_ICG_PD_3_6_LED_D',
           'Concentration_HbO_PD_4_5_LED_D','Concentration_HbR_PD_4_5_LED_D','Concentration_CtOx_PD_4_5_LED_D','Concentration_ICG_PD_4_5_LED_D']    
df = pd.read_csv('filter/Concentration_raw_voltage_data_HbR_HbO_Kolyva_6.5µM_ICG_Landesman.csv', names=headers, dtype=str)
headers=np.array(headers)

N=len(headers)-1
N=int(N/4)
M=4
data=np.asarray(df)
data=(data[1:,:])
data = data = data.astype('float')
(M,N)=data.shape


subtitle=['StO2_PD_1_4_LED_R',
            'StO2_PD_2_3_LED_R',
            'StO2_PD_1_4_LED_L',
            'StO2_PD_2_3_LED_L',
            'StO2_PD_5_8_LED_R',
            'StO2_PD_6_7_LED_R',
            'StO2_PD_5_8_LED_L',
            'StO2_PD_6_7_LED_L',
            'StO2_PD_1_8_LED_U',
            'StO2_PD_2_7_LED_U',
            'StO2_PD_3_6_LED_U',
            'StO2_PD_4_5_LED_U',
            'StO2_PD_1_8_LED_D',
            'StO2_PD_2_7_LED_D',
            'StO2_PD_3_6_LED_D',
            'StO2_PD_4_5_LED_D']
K=N/4
subcol=4
subrow=int(K/4)
subplot_row_col=np.arange(1,K+1).reshape([int(K/4),-1])
(subplot_row,subplot_col)=subplot_row_col.shape

fig = make_subplots(
    rows=subrow, cols=subcol,
    subplot_titles=subtitle)

subtitle=np.reshape(subtitle,([subrow,subcol]))


xs=np.arange(0,M)
xs=xs/3


data_reshaped=data.reshape([-1,int(N/4),4])
data_set=data_reshaped[:,1,1]
#kalman_filtered_signal=np.zeros(len(xs))
tHbO=data_reshaped[:,:,0]
tHbR=data_reshaped[:,:,1]
tHb=tHbO+tHbR
StO2=(tHbO/tHb)*100

StO2_reshaped=StO2.reshape([-1,int(K/4),4])
kalman_filtered_signal=np.zeros(len(xs))


for i in range(0, subplot_row):
    for j in range(0, subplot_col):
        ys=StO2_reshaped[:,i,j]
  
        
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
            
        fig.add_trace(go.Scatter(x=xs,y=ys,
                     mode='lines',line_color="crimson",
                     line=dict(width=1),
                     name=subtitle[i,j]),
           row=i+1,col=j+1)

        fig.add_trace(go.Scatter(x=xs,y=kalman_filtered_signal,
                     mode='lines',line_color="gray",
                     line=dict(width=1),
                     name=subtitle[i,j]+'_Kalman_filter',),
           row=i+1,col=j+1)
        
        fig.add_trace(go.Scatter(x=xs,y = Lowpass_filtered_signal,
                     mode='lines',line_color="black",
                     line =dict(width=1),
                    name=subtitle[i,j]+'_Lowpass_filter'),
           row=i+1,col=j+1)
        fig.update_xaxes(title_text="Time in s", row=i+1, col=j+1)
        fig.update_yaxes(title_text="StO_2 in %", row=i+1, col=j+1)   
fig.update_layout(width=2100,height=1500,title_text="Enable Pig 05 ",plot_bgcolor="rgb(256,256,256)")

plot(fig,filename="StO2_Concentration_raw_voltage_data.html")
fig.show()
fig.write_image("images/StO2_Concentration_raw_voltage_data.svg")
fig.write_html("html/StO2_Concentration_raw_voltage_data.html")