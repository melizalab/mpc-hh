import sys
sys.path.append('neuron_scripts')
sys.path.append('DDF_scripts')
from DDF import *
import numpy as np
import argparse
import matplotlib.pyplot as plt
from connor_stevens import config_dict
from waveform_analysis import *

# ============================
# Parse Command Line Arguments
# ============================
p = argparse.ArgumentParser()
p.add_argument('--data_file', default='assimilation_data/type_I.npy', type=str, help='file path to assimilation data')
p.add_argument('--config_file', default='config/config_cs_type_I.yaml', type=str, help='file path to config file for assimilation data')
p.add_argument('--ddf_path', default = 'saved_models/ddf_type_I.npy', type = str, help = 'path to DDF model.')
args = p.parse_args()

# ==============
# Load DDF Model
# ==============
ddf_data = np.load(args.ddf_path,allow_pickle=True)[()]
coeffs,centers,D_e,tau,scale = ddf_data['Model parameters'].values()

# =================
# Get Time Metadata
# =================
'''
dsr: downsample rate
delta_t: time step size of downsampled data
t0,tn: initial and final times
'''
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
fig,ax = plt.subplots(2,1,sharex=True,sharey=True)

# Plot Actual vs Predicted
ax[0].vlines(actual_spikes,60,65,color='darkcyan')
ax[0].vlines(ddf_spikes,70,75,color='black')
ax[0].plot(t,V,color='darkcyan',alpha=0.7)
ax[0].plot(t,V_hat,color='black',alpha=0.7)

# Plot Perfect Model vs Predicted
ax[1].vlines(ref_spikes,60,65,color='darkcyan')
ax[1].vlines(ddf_spikes,70,75,color='black')
ax[1].plot(t,ref_V,color='darkcyan',alpha=0.7)
ax[1].plot(t,V_hat,color='black',alpha=0.7)
plt.show()

# =================
# Plot Spike Trains
# =================
plt.vlines(ref_spikes,0,0.8,color='darkcyan')
plt.vlines(ddf_spikes,1,1.8,color='black')
for trial in range(n_trials):
    plt.vlines(comp_spikes['spike times'][trial],0+(trial+3),0.8+(trial+3),color='blue')
plt.show()
