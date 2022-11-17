# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:25:02 2022

@author: hmchayme
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:48:19 2022

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
    
headers = ['\
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
df = pd.read_csv('filter/Concentration_raw_voltage_data_HbR_HbO_Gratzer_6.5µM_ICG_Landesman.csv', dtype=str)
headers=np.array(headers)

N=len(headers)-1
N=int(N/4)
M=4
data=df.to_numpy(dtype=float)
data=data[:,3:]
N=64
#data=data_array.astype(float)

#data=np.array(K[:,:],dtype=(float))

(M,N)=data.shape

subcol=3
subrow=int(N/3)
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
    

fig = make_subplots(
    rows=subrow, cols=subcol,
    subplot_titles=subtitle)

subtitle=np.reshape(subtitle,([subrow,subcol]))


xs=np.arange(0,M)
xs=xs/3

subplot_row_col=np.arange(1,N+1).reshape([int(N/3),-1])
(subplot_row,subplot_col)=subplot_row_col.shape
data_reshaped=data.reshape([-1,int(N/3),3])
data_set=data_reshaped[:,1,1]
kalman_filtered_signal=np.zeros(len(xs))


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
            
        fig.add_trace(go.Scatter(x=xs,y=ys,
                     mode='lines',line_color="crimson",
                     line=dict(width=1),
                     name=subtitle[i,j]),
                  row=i+1,col=j+1)
        
        fig.add_trace(go.Scatter(x=xs,y = Lowpass_filtered_signal,
                     mode='lines',line_color="black",
                     line =dict(width=1),
                    name=subtitle[i,j]+'_Lowpass_filter'),
                  row=i+1,col=j+1)
        fig.update_xaxes(title_text="Time in s", row=i+1, col=j+1)
        fig.update_yaxes(title_text="Concentration in mM", row=i+1, col=j+1)   
        


fig.update_layout(width=2100,height=4000,title_text="Enable Pig 05 ",plot_bgcolor="rgb(256,256,256)")

plot(fig,filename="Concentration_raw_voltage_data.html")
fig.show()
fig.write_image("images/Concentration_raw_voltage_data.svg")
fig.write_html("html/Concentration_raw_voltage_data.html")