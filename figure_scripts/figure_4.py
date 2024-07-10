import sys
sys.path.append('neuron_scripts')
sys.path.append('DDF_scripts')
from DDF import *
import numpy as np
import argparse
import matplotlib.pyplot as plt
from connor_stevens import config_dict
from waveform_analysis import *
import pyspike as spk


'''
Creates plots for Figure 4.
To rerun code, you will need to have a path to the assimilation data and ddf_model.
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

# ============================
# Parse Command Line Arguments
# ============================
p = argparse.ArgumentParser()
p.add_argument('--data_file', default='assimilation_data/type_II.npy', type=str, help='file path to assimilation data')
p.add_argument('--config_file', default='config/config_cs_type_II.yaml', type=str, help='file path to config file for assimilation data')
p.add_argument('--ddf_path', default = 'saved_models/ddf_type_II.npy', type = str, help = 'path to DDF model.')
args = p.parse_args()

# ==============
# Load DDF Model
# ==============
ddf_data = np.load(args.ddf_path,allow_pickle=True)[()]
coeffs,centers,D_e,tau,scale = ddf_data['Model parameters'].values()
dsr,delta_t,t0,tn = ddf_data['Time'].values()
t = np.arange(t0,tn,delta_t)

# ====================
# Load Validation Data
# ====================
param_dict = config_dict(args.config_file)
assimilation_data = np.load(f'{args.data_file}',allow_pickle=True)[()]
sim_type = param_dict['Name']['sim type']
# Validation Data
V = assimilation_data['Validation']['V'][::dsr]
actual_spikes = rasterize(t,V)
I_inj = assimilation_data['Validation']['I_inj'][::dsr]
color_dict = {'I':'darkcyan','II':'darkmagenta'}

# ====================================
# Load Reference and Comparison Spikes
# ====================================
ref_data = np.load(f'open_loop_scripts/type_{sim_type}_ref.npy',allow_pickle=True)[()]
ref_spikes = ref_data['spike']
ref_V = ref_data['V'][::dsr]
comp_spikes = np.load(f'open_loop_scripts/null_spikes_{sim_type}.npy',allow_pickle=True)[()]
n_trials = comp_spikes['n trials']

# =========================
# Use DDF Model to Forecast
# =========================
V_hat = predict_future(V,I_inj,tau,D_e,centers,coeffs,scale)
ddf_spikes = rasterize(t,V_hat)

# ================
# Plot Comparisons
# ================
fig,ax = plt.subplots(1,3,sharex=True,figsize=(20,3))
# Plot Actual vs Predicted
ax[0].vlines(actual_spikes,60,65,color=color_dict[sim_type],linewidth=1)
ax[0].vlines(ddf_spikes,70,75,color='black',linewidth=1)
ax[0].plot(t,V,color=color_dict[sim_type],alpha=0.7,linewidth=1)
ax[0].plot(t,V_hat,color='black',alpha=0.7,linewidth=1)
ax[0].set_xlim([-40,2040])
ax[0].set_xticks([0,1000,2000])

ax[1].plot(t,assimilation_data['Validation']['I_inj'][::dsr],color='black',alpha=0.7,linewidth=1)
ax[1].plot(t,assimilation_data['Validation']['I_noise'][::dsr],color='#ff7f0e',alpha=0.7,linewidth=1)
ax[1].set_xlim([-40,2040])
ax[1].set_xticks([0,1000,2000])

# Spike Train Similarities
ddf_spike_train = spk.SpikeTrain(ddf_spikes, [0, 5000])
actual_spike_train = spk.SpikeTrain(actual_spikes, [0, 5000])
ISI = spk.isi_distance(ddf_spike_train, actual_spike_train, interval=(0, 5000))
spike = spk.spike_distance(ddf_spike_train, actual_spike_train, interval=(0, 5000))
print('Forecast:')
print(f'----> ISI Distance:{ISI:.2f}')
print(f'----> Spike Distance:{spike:.2f}')


# =================
# Plot Spike Trains
# =================
ax[2].vlines(ref_spikes,31,31.8,color=color_dict[sim_type],linewidth=1)
ax[2].vlines(ddf_spikes,32,32.8,color='black',linewidth=1)
for trial in range(n_trials):
    ax[2].vlines(comp_spikes['spike times'][trial],0+trial,0.8+trial,color='darkgoldenrod',linewidth=1)
ax[2].set_xlim([-40,2040])
ax[2].set_xticks([0,1000,2000])
#plt.savefig('figure_parts/figure_4_a.pdf')
plt.show()

# Spike train similarities
ddf_spike_train = spk.SpikeTrain(ddf_spikes, [0, 5000])
ref_spike_train = spk.SpikeTrain(ref_spikes, [0, 5000])
ISI = spk.isi_distance(ddf_spike_train, ref_spike_train, interval=(0, 5000))
spike = spk.spike_distance(ddf_spike_train, ref_spike_train, interval=(0, 5000))
print('Perfect Model')
print(f'----> ISI Distance:{ISI:.2f}')
print(f'----> Spike Distance:{spike:.2f}')
