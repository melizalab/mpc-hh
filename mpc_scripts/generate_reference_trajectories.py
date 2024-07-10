import sys
sys.path.append('neuron_scripts')
import numpy as np
import argparse
from stimulate_neuron import stimulate
from neuron_inputs import chaotic_current,balanced_noise_current,scale_noise_to_SNR
from connor_stevens import config_dict

# =======================
# Parse Command Line Args
# =======================
p = argparse.ArgumentParser()
p.add_argument('--config_file', default = 'config/config_cs_type_I.yaml', type = str, help = 'config file path for data generation.')
p.add_argument('--n_trials',default=50,type=int,help='number of reference trajectories to make')
p.add_argument('--len_of_trial',default=500,type=int,help='length of time for trajectory (in ms)')
p.add_argument('--out_path', default = 'reference_trajectories', type = str, help = 'file path to save output of simulation.')
args = p.parse_args()

# =========================================================
# Read Config File to Get Assimilation Data Hyperparameters
# =========================================================
param_dict = config_dict(args.config_file)
sim_type = param_dict['Name']['sim type']

# =====================
# Construct Time Vector
# =====================
_,_,dt = param_dict['Time'].values()
t = np.arange(0,args.len_of_trial,dt)

# ==============================
# Get I_inj and I_noise Metadata
# ==============================
input_scaling,tau_inj = param_dict['Injected Current'].values()
snr,tau_noise,g_syn_max,spiker_fr = param_dict['Noise Current'].values()

# ================================
# Construct Reference Trajectories
# ================================
Vs = np.empty((args.n_trials,len(t))) # membrane voltages
I_injs = np.empty_like(Vs) # injected currents
I_noises = np.empty_like(Vs) # noise currents
for trial in range(args.n_trials):
    print(f'Trial: {trial}')
    I_inj= chaotic_current(0,args.len_of_trial, dt, input_scaling, tau_inj)
    I_noise = balanced_noise_current(spiker_fr, dt, t, g_syn_max, tau_noise)
    I_noise = scale_noise_to_SNR(I_inj, I_noise, snr)
    V = stimulate(t, I_inj, I_noise, args.config_file,monitor=True)
    # Append data
    Vs[trial] = V
    I_injs[trial] = I_inj
    I_noises[trial] = I_noise

# =========
# Save data
# =========
data_dict = {'V':Vs,'I_inj':I_injs,'I_noise':I_noises}
np.save(f'{args.out_path}/type_{sim_type}_ref_trajectories.npy',data_dict)



