import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('neuron_scripts')
from waveform_analysis import rasterize


'''
Creates the plots for Figure 8
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

# ========
# MPC Data
# ========
# Trial indexes to use
type_I_indx = 0
type_II_indx = 32
'''
exp_3 originally had many trials of spike train control, but only one was used for 
revised manuscript
'''
exp_num = 'exp_3'
# Only plot first 300 ms for clarity of figure
t0,tn = 0,300
dt = .02
delta_t = 0.1
dsr = 5
t = np.arange(t0,tn,dt)
T = np.arange(t0,tn,delta_t)
# Get Data Type I
mpc_data_type_I = np.load(f'control_output/{exp_num}/type_I_trial_indx_{type_I_indx}.npy',allow_pickle=True)[()]
# Get Data Type II
mpc_data_type_II = np.load(f'control_output/{exp_num}/type_II_trial_indx_{type_II_indx}.npy',allow_pickle=True)[()]

# =================
# P Controller Data
# =================
# Get Data Type I
p_data_type_I = np.load(f'p_control_output/{exp_num}/type_I_trial_indx_{type_I_indx}.npy',allow_pickle=True)[()]
# Get Data Type II
p_data_type_II = np.load(f'p_control_output/{exp_num}/type_II_trial_indx_{type_II_indx}.npy',allow_pickle=True)[()]

# =================
# Pulse Controller Data
# =================
# Get Data Type I
pulse_data_type_I = np.load(f'open_loop_output/{exp_num}/type_I_trial_indx_{type_I_indx}.npy',allow_pickle=True)[()]
# Get Data Type II
pulse_data_type_II = np.load(f'open_loop_output/{exp_num}/type_II_trial_indx_{type_II_indx}.npy',allow_pickle=True)[()]


def fig_function(type_I_data,type_II_data,col,t,T,y_ticks,pulse_flag=False):
    ######## Type I States
    if pulse_flag == False:
        ax[0, col].plot(T, type_I_data['reference trajectory'][t0:len(T)], color='black', alpha=0.7,linewidth=1)  # Ref Traj
        ax[0, col].vlines(rasterize(T, type_I_data['reference trajectory'][t0:len(T)]), 60, 70, color='black',alpha=0.7,linewidth=1)  # Ref Spikes
    else:
        ax[0, col].plot(t, type_I_data['reference trajectory'][t0:len(t)], color='black', alpha=0.7,linewidth=1)  # Ref Traj
        ax[0, col].vlines(rasterize(t, type_I_data['reference trajectory'][t0:len(t)]), 60, 70, color='black',alpha=0.7,linewidth=1)  # Ref Spikes
    ax[0, col].plot(t, type_I_data['V_actual'][t0:len(t)], color='red', alpha=0.7,linewidth=1)  # MPC Controlled States
    ax[0, col].vlines(rasterize(t, type_I_data['V_actual'][t0:len(t)]), 75, 85, color='red',alpha=0.7,linewidth=1)  # MPC Control Spikes
    ax[0,col].set_yticks(y_ticks[0])
    ######## Type I Inputs
    ax[1, col].plot(t, type_I_data['I_control'][t0:len(t)], alpha=0.7,linewidth=1)
    ax[1, col].plot(t, type_I_data['I_noise'][t0:len(t)], alpha=0.7,linewidth=1)
    ax[1, col].set_yticks(y_ticks[1])
    ######## Type II States
    if pulse_flag ==False:
        ax[2, col].plot(T, type_II_data['reference trajectory'][t0:len(T)], color='black', alpha=0.7,linewidth=1)
        ax[2, col].vlines(rasterize(T, type_II_data['reference trajectory'][t0:len(T)]), 60, 70, color='black',alpha=0.7,linewidth=1)
    else:
        ax[2, col].plot(t, type_II_data['reference trajectory'][t0:len(t)], color='black', alpha=0.7,linewidth=1)
        ax[2, col].vlines(rasterize(t, type_II_data['reference trajectory'][t0:len(t)]), 60, 70, color='black',alpha=0.7,linewidth=1)
    ax[2, col].plot(t, type_II_data['V_actual'][t0:len(t)], color='red', alpha=0.7,linewidth=1)
    ax[2, col].vlines(rasterize(t, type_II_data['V_actual'][t0:len(t)]), 75, 85, color='red', alpha=0.7,linewidth=1)
    ax[2, col].set_yticks(y_ticks[2])
    ######## Type I Inputs
    ax[3, col].plot(t, type_II_data['I_control'][:len(t)], alpha=0.7,linewidth=1)
    ax[3, col].plot(t, type_II_data['I_noise'][:len(t)], alpha=0.7,linewidth=1)
    ax[3, col].set_yticks(y_ticks[3])


# ==================
# Figure Making Time
# ==================
# Columns for each controller
mpc_col = 0
p_col = 1
pulse_col = 2
# Plot figure
fig,ax = plt.subplots(4,3,sharex=True,figsize = (20,7))
fig_function(mpc_data_type_I,mpc_data_type_II,mpc_col,t,T,y_ticks=[[-100,0,100],[-25,0,25],[-100,0,100],[-25,0,25]])
fig_function(p_data_type_I,p_data_type_II,p_col,t,T,y_ticks=[[-100,0,100],[-100,0,100],[-100,0,100],[-100,0,100]])
fig_function(pulse_data_type_I,pulse_data_type_II,pulse_col,t,T,pulse_flag=True,y_ticks=[[-100,0,100],[0,50,100],[-100,0,100],[0,25,50]])
plt.tight_layout()
#plt.savefig('figure_parts/figure_8.pdf')
plt.show()

