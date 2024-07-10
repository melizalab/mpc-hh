import numpy as np
import matplotlib.pyplot as plt

'''
Creates plots for Figure 6b (exp_num = 'exp_1') and Figure 7b (exp_num='exp_2')
To rerun code, you will need to have a path to the control outputs.
'''

###################
# Plotting Params #
###################
fontsize = 16
tick_thickness = 1 #3
tick_length = 8
plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize,
                     'xtick.major.width': tick_thickness, 'ytick.major.width': tick_thickness,
                     'xtick.major.size': tick_length, 'ytick.major.size': tick_length})


exp_num = 'exp_2'

# MPC Data
mpc_data_type_I = np.load(f'control_output/mpc_performance_type_I_{exp_num}.npy',allow_pickle=True)[()]
mpc_data_type_II = np.load(f'control_output/mpc_performance_type_II_{exp_num}.npy',allow_pickle=True)[()]
# Open Loop data
open_data_type_I = np.load(f'open_loop_output/open_loop_performance_type_I_{exp_num}.npy',allow_pickle=True)[()]
open_data_type_II = np.load(f'open_loop_output/open_loop_performance_type_II_{exp_num}.npy',allow_pickle=True)[()]

metric = 'SpkDs'
plt.figure(figsize=(3,7))
plt.boxplot([mpc_data_type_I[f'{metric}'],open_data_type_I[f'{metric}'],mpc_data_type_II[f'{metric}'],open_data_type_II[f'{metric}']],positions=[1,1.5,3,3.5])
plt.xticks([1.25,3.25])
#plt.savefig(f'figure_parts/fig_7_c_{metric}.pdf')
plt.show()