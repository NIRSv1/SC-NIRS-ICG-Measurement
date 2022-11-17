# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:58:21 2022

@author: hmchayme
"""

""" -*- coding: utf-8 -*-

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
from numpy.fft import fft, ifft
if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("Data"):
    os.mkdir("Data")
if not os.path.exists("filter"):
    os.mkdir("filter")    
if not os.path.exists("html"):
    os.mkdir("html")
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

# %% Variable definition
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
#Moaveni		and Landesman
citation='_HbR_HbO_Gratzer_6.5µM_ICG_Landesman'
epsylon_690_HbO=0.031222
epsylon_690_HbR=0.213764
epsylon_690_CtOx=0.263213
epsylon_690_ICG=2.8169 #6.5µM

epsylon_780_HbO=0.073583
epsylon_780_HbR=0.110471
epsylon_780_CtOx=0.204948
epsylon_780_ICG=11.511 #6.55µM

epsylon_810_HbO=0.092887
epsylon_810_HbR=0.079849
epsylon_810_CtOx=0.231661
epsylon_810_ICG=19.072 #´6.5µM

epsylon_850_HbO=0.115931
epsylon_850_HbR=0.07859
epsylon_850_CtOx=0.22899
epsylon_850_ICG=2.9618 #6.5µM
#epsylon_lambda_chromophore=[   epsylon_690_HbO epsylon_690_HbR epsylon_690_CtOx  epsylon_690_ICG
#                               epsylon_780_HbO epsylon_780_HbR epsylon_780_CtOx  epsylon_780_ICG
#                               epsylon_810_HbO epsylon_810_HbR epsylon_810_CtOx  epsylon_810_ICG
#                               epsylon_850_HbO epsylon_850_HbR epsylon_850_CtOx  epsylon_850_ICG]


epsylon_690_780_810_850_HbO_HbR_CtOx_ICG=np.array([[epsylon_690_HbO, epsylon_690_HbR, epsylon_690_CtOx,  epsylon_690_ICG],
                                                          [epsylon_780_HbO, epsylon_780_HbR, epsylon_780_CtOx,  epsylon_780_ICG],
                                                          [epsylon_810_HbO, epsylon_810_HbR, epsylon_810_CtOx,  epsylon_810_ICG],
                                                          [epsylon_850_HbO, epsylon_850_HbR, epsylon_850_CtOx,  epsylon_850_ICG]])
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
headers = ['TStamp','Trg ','Frame ',
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
df = pd.read_csv('filter/Concentration_raw_voltage_data_HbR_HbO_Gratzer_6.5µM_ICG_Landesman.csv', dtype=str)
headers=np.array(headers)
df_2=df[['TStamp',' Trg',' Frame']]

data=np.asarray(df)
sr = 3
# sampling interval
ts = 1.0/sr
ts = 1.0/sr

# %% The voltages  in 2D matrix

data = data = data.astype('float')
data_voltages_raw=data[:,3:]
(M,N)=data_voltages_raw.shape
N=N

subcol=4
subrow=int(N/4)
subtitle=['\
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
HbO_S02D04D05S03','HbR_S02D04D05S03','ICG_S02D04D05S03']
#%% The voltages  in 3D matrix

#%%

fig = make_subplots(
    rows=subrow, cols=subcol,
    subplot_titles=subtitle)
subtitle=np.reshape(subtitle,([subrow,subcol]))


xs=np.arange(0,M)
xs=xs/3
subplot_row_col=np.arange(1,N+1).reshape([int(N/4),-1])
(subplot_row,subplot_col)=subplot_row_col.shape
data_reshaped=data_voltages_raw.reshape([-1,int(N/4),4])
data_set=data_reshaped[:,1,1]



Lowpass_filtered_Voltage_data=np.zeros((M,1))

            
#%%the ploted filterd voltage data

for i in range(0, subplot_row):
    for j in range(0, subplot_col):
        x=data_reshaped[:,i,j]

        ys=x.reshape(len(x))
        
        '''
        freq=1
        x +=10 *np.sin(2*np.pi*freq*t)
        freq=0.5
        x +=10 *np.sin(2*np.pi*freq*t)
        '''
        
        x1=x-np.average(x)
        x2 = (x - x.mean()) / (x.max() - x.min())
        N = x2 .size
        
        #x=x[:48]
        t = np.arange(0,len(x)/sr,ts)
        n = np.arange(N)
        T = N/sr
        freq = n/T 
        n_oneside = N//2
        # get the one side frequency
        f_oneside = freq[:n_oneside]
        X=fft(x2)
        fig.add_trace(go.Scatter(x=f_oneside, y=np.abs(X[:n_oneside]),
                     mode='lines',line_color="crimson",
                     line=dict(width=1),
                     name=subtitle[i,j]),                    
           row=i+1,col=j+1)


        fig.update_xaxes(title_text="Frequency in Hz", row=i+1, col=j+1)
        fig.update_yaxes(title_text="Amplitude", row=i+1, col=j+1)  
fig.update_layout(width=1700,height=5000,title_text="FFT raw volatge",plot_bgcolor="rgb(256,256,256)")#,xaxis=dict(tickvals=xs/3))

plot(fig, filename="Raw_voltage_fft.html")
fig.show()
fig.write_image("images/Raw_voltage_fft.svg")
fig.write_html("html/Raw_voltage_fft.html")


