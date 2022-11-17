
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import normalize
from matplotlib.ticker import NullFormatter

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

df = pd.read_csv('filter/Concentration_raw_voltage_data_HbR_HbO_Gratzer_6.5µM_ICG_Landesman.csv')

data=np.asarray(df)

data = data.astype('float')
data_voltages_raw=data[:,3:]
data_voltages_raw=data_voltages_raw
n_LED_x_PD_y=data_voltages_raw.reshape(-1,16,3)
n_LED_x_PD_y=n_LED_x_PD_y*10**3
n_LED_x_PD_y=n_LED_x_PD_y[100:-20,:,:]
(M,N)=data_voltages_raw.shape
xs=np.arange(0,M-120)
xs=xs/3


fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex=True,figsize=(14,10))

HbO_S04D02D03S03=n_LED_x_PD_y[:,7,0]
HbR_S04D02D03S03=n_LED_x_PD_y[:,7,1]
tHb_S04D02D03S03=n_LED_x_PD_y[:,7,0]+n_LED_x_PD_y[:,7,1]
ICG_S04D02D03S03=n_LED_x_PD_y[:,7,2]
print('HbO')
print(np.mean(HbO_S04D02D03S03))
print(np.std(HbO_S04D02D03S03))
print('HbR')
print(np.mean(HbR_S04D02D03S03))
print(np.std(HbR_S04D02D03S03))
print('tHb')
print(np.mean(tHb_S04D02D03S03))
print(np.std(tHb_S04D02D03S03))
print('ICG_max')
print(np.max(ICG_S04D02D03S03)-np.mean(ICG_S04D02D03S03[0:20]))

ax1.plot(xs, HbO_S04D02D03S03,color='gray')
ax1.set_title('[HbO]')
ax2.plot(xs, HbR_S04D02D03S03,color='gray')
ax2.set_title('kHbR]')
ax3.plot(xs, tHb_S04D02D03S03,color='gray')
ax3.set_title('[tHb]')
ax4.plot(xs, ICG_S04D02D03S03-np.mean(ICG_S04D02D03S03[0:20]),color='gray')
ax4.set_title('[ICG]')
df = pd.read_csv('filter/Concentration_Lowpass_filtered_voltage_data _HbR_HbO_Gratzer_6.5µM_ICG_Landesman.csv')

data=np.asarray(df)

data = data.astype('float')
data_voltages_raw=data[:,3:]
data_voltages_raw=data_voltages_raw
n_LED_x_PD_y=data_voltages_raw.reshape(-1,16,3)
n_LED_x_PD_y=n_LED_x_PD_y*10**3
n_LED_x_PD_y=n_LED_x_PD_y[100:-20,:,:]
(M,N)=data_voltages_raw.shape
xs=np.arange(0,M-120)
xs=xs/3


#fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1)

HbO_S04D02D03S03=n_LED_x_PD_y[:,7,0]
HbR_S04D02D03S03=n_LED_x_PD_y[:,7,1]
tHb_S04D02D03S03=n_LED_x_PD_y[:,7,0]+n_LED_x_PD_y[:,7,1]
ICG_S04D02D03S03=n_LED_x_PD_y[:,7,2]
print('HbO')
print(np.mean(HbO_S04D02D03S03))
print(np.std(HbO_S04D02D03S03))
print('HbR')
print(np.mean(HbR_S04D02D03S03))
print(np.std(HbR_S04D02D03S03))
print('tHb')
print(np.mean(tHb_S04D02D03S03))
print(np.std(tHb_S04D02D03S03))
print('ICG_max')
print(np.max(ICG_S04D02D03S03)-np.mean(ICG_S04D02D03S03[0:20]))
print(np.max(ICG_S04D02D03S03)-np.mean(ICG_S04D02D03S03[0:20]))

ax1.plot(xs, HbO_S04D02D03S03,color='#ff2a2aff')
ax1.grid(True)
#ax1.set_yscale('symlog')
ax2.plot(xs, HbR_S04D02D03S03,color='#ff2a2aff')
ax2.grid(True)
#ax2.set_yscale('symlog')
ax3.plot(xs, tHb_S04D02D03S03,color='#ff2a2aff')
ax3.grid(True)
#ax3.set_yscale('symlog')
ax4.plot(xs, ICG_S04D02D03S03-np.mean(ICG_S04D02D03S03[0:20]),color='#ff2a2aff')
ax4.grid(True)
ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.02f}'))
ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.03f}'))
#ax4.set_yscale('symlog')

'''
ax1.plot(xs, HbO_S04D02D03S03,color='#c7564dff')
ax1.set_title('[HbO]')
ax2.plot(xs, HbR_S04D02D03S03,color='#00355dff')
ax2.set_title('[HbR]')
ax3.plot(xs, tHb_S04D02D03S03,color='#90715eff')
ax3.set_title('[tHb]')
ax4.plot(xs, ICG_S04D02D03S03,color='#909b5eff')
ax4.set_title('[ICG]')

ax2.plot(xs, n_LED_13_PD_3,color='#ff8080ff')
ax2.plot(xs,n_LED_14_PD_3,color='#ff2a2aff')
ax2.plot(xs, n_LED_15_PD_3,color='#bb0000')
ax2.plot(xs, n_LED_16_PD_3,color='#800000ff')

ax3.plot(xs, n_LED_9_PD_2,color='#800000ff')
ax3.plot(xs,n_LED_10_PD_2,color='#bb0000')
ax3.plot(xs, n_LED_11_PD_2,color='#ff2a2aff')
ax3.plot(xs, n_LED_12_PD_2,color='#ff8080ff')

ax4.plot(xs, n_LED_9_PD_3,color='#800000ff')
ax4.plot(xs,n_LED_10_PD_3,color='#bb0000')
ax4.plot(xs, n_LED_11_PD_3,color='#ff2a2aff')
ax4.plot(xs, n_LED_12_PD_3,color='#ff8080ff')
'''
ax2.set_ylabel('Absolute concentration in µM')
ax4.set_xlabel('Time in s')
fig.tight_layout()

fig.legend(['Raw ', 'Filtered'],bbox_to_anchor =(1.0, 1.02))#,ncol=7)



#fig.text(-0.0, 0.5, 'Absolute concentration in \microM', va='center', rotation='vertical',fontsize=25)
#fig.text(0.5,-0.03, 'Time in s', va='center',fontsize=15)
plt.rc('axes', titlesize=30) #fontsize of the title
plt.rc('axes', labelsize=30) #fontsize of the x and y labels
plt.rc('xtick', labelsize=25) #fontsize of the x tick labels
plt.rc('ytick', labelsize=25) #fontsize of the y tick labels
plt.rc('legend', fontsize=27) #fontsize of the legend

fig.tight_layout()
fig.savefig("LEDset3_4_PD6_7_Concentration.svg")
plt.show()
