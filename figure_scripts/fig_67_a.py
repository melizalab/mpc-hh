import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('neuron_scripts')
from waveform_analysis import rasterize

'''
Creates plots for Figure 6a (exp_num = 'exp_1') and Figure 7a (exp_num='exp_2')
To rerun code, you will need to have a path to the control outputs.
'''

###################
# Plotting Params #
###################
fontsize = 16
tick_thickness = 1
tick_length = 8
plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize,
                     'xtick.major.width': tick_thickness, 'ytick.major.width': tick_thickness,
                     'xtick.major.size': tick_length, 'ytick.major.size': tick_length})

exp_num = 'exp_1'
trial_indx_type_I = 0
trial_indx_type_II = 32

# Time
t = np.arange(0,500,0.02)
T = np.arange(0,500,.1)

# Get Data Type I
open_data_type_I = np.load(f'open_loop_output/{exp_num}/type_I_trial_indx_{trial_indx_type_I}.npy',allow_pickle=True)[()]
mpc_data_type_I = np.load(f'control_output/{exp_num}/type_I_trial_indx_{trial_indx_type_I}.npy',allow_pickle=True)[()]

# Get Data Type II
open_data_type_II = np.load(f'open_loop_output/{exp_num}/type_II_trial_indx_{trial_indx_type_II}.npy',allow_pickle=True)[()]
mpc_data_type_II = np.load(f'control_output/{exp_num}/type_II_trial_indx_{trial_indx_type_II}.npy',allow_pickle=True)[()]


fig,ax = plt.subplots(3,2,sharex=True,figsize=(10,5))

# Open Loop References
ax[0,0].plot(t,open_data_type_I['reference trajectory'],color='black',alpha=0.7)
ax[0,1].plot(t,open_data_type_II['reference trajectory'],color='black',alpha=0.7)

# Open Loop Ref Spikes
ax[0,0].vlines(rasterize(t,open_data_type_I['reference trajectory']),60,70,color='black',alpha=0.7)
ax[0,1].vlines(rasterize(t,open_data_type_II['reference trajectory']),60,70,color='black',alpha=0.7)

# Open Loop Controlled States
ax[0,0].plot(t,open_data_type_I['V_actual'],color='red',alpha=0.7)
ax[0,1].plot(t,open_data_type_II['V_actual'],color='red',alpha=0.7)

# Open Loop Control Spikes
ax[0,0].vlines(rasterize(t,open_data_type_I['V_actual']),75,85,color='red',alpha=0.7)
ax[0,1].vlines(rasterize(t,open_data_type_II['V_actual']),75,85,color='red',alpha=0.7)
#====================================================================
# MPC References

ax[1,0].plot(T,mpc_data_type_I['reference trajectory'][:len(T)],color='black',alpha=0.7)
ax[1,1].plot(T,mpc_data_type_II['reference trajectory'][:len(T)],color='black',alpha=0.7)

# MPC Ref Spikes
ax[1,0].vlines(rasterize(T,mpc_data_type_I['reference trajectory'][:len(T)]),60,70,color='black',alpha=0.7)
ax[1,1].vlines(rasterize(T,mpc_data_type_II['reference trajectory'][:len(T)]),60,70,color='black',alpha=0.7)

# MPC Controlled States
ax[1,0].plot(t,mpc_data_type_I['V_actual'][:len(t)],color='red',alpha=0.7)
ax[1,1].plot(t,mpc_data_type_II['V_actual'][:len(t)],color='red',alpha=0.7)

# MPC Control Spikes
ax[1,0].vlines(rasterize(t,mpc_data_type_I['V_actual'][:len(t)]),75,85,color='red',alpha=0.7)
ax[1,1].vlines(rasterize(t,mpc_data_type_II['V_actual'][:len(t)]),75,85,color='red',alpha=0.7)

#====================================================================

# Open Loop Inputs
ax[2,0].plot(t,open_data_type_I['I_control'],alpha=0.7,color='black')
ax[2,1].plot(t,open_data_type_II['I_control'],alpha=0.7,color='black')

# MPC Inputs
ax[2,0].plot(t,mpc_data_type_I['I_control'],alpha=0.7)
ax[2,1].plot(t,mpc_data_type_II['I_control'],alpha=0.7)

# Noise Inputs
ax[2,0].plot(t,open_data_type_I['I_noise'],alpha=0.7)
ax[2,1].plot(t,open_data_type_II['I_noise'],alpha=0.7)

# plotting crap
ax[0,0].set_yticks([-100,0,100])
ax[1,0].set_yticks([-100,0,100])
ax[0,1].set_yticks([-100,0,100])
ax[1,1].set_yticks([-100,0,100])

ax[0,0].set_ylim([-110,110])
ax[1,0].set_ylim([-110,110])
ax[0,1].set_ylim([-110,110])
ax[1,1].set_ylim([-110,110])

ax[2,0].set_xticks([0,250,500])
ax[2,1].set_xticks([0,250,500])

#plt.savefig('figure_parts/fig_6_ab.pdf')
plt.show()
