import sys
sys.path.append('neuron_scripts')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from stimulate_neuron import stimulate
from neuron_inputs import chaotic_current,balanced_noise_current,scale_noise_to_SNR
from connor_stevens import config_dict
import argparse
from waveform_analysis import rasterize

'''
Creates spikes for Figure 4. Repeated stimulation of the CS neurons to different I_noise.
'''

# =======================
# Parse Command Line Args
# =======================
p = argparse.ArgumentParser()
p.add_argument('--config_file', default = 'config/config_cs_type_II.yaml', type = str, help = 'config file path for data generation.')
p.add_argument('--data_file', default='assimilation_data/type_II.npy', type=str, help='file path to assimilation data')
p.add_argument('--n_trials', default=30,type=int)
p.add_argument('--DSR', default=5, type=int, help='downsample rate of model (i.e. 5 -> only use every 5 measurements')
p.add_argument('--out_path', default = 'open_loop_scripts', type = str, help = 'file path to save output of simulation.')
args = p.parse_args()

# =========================================================
# Read Config File to Get Assimilation Data Hyperparameters
# =========================================================
param_dict = config_dict(args.config_file)
sim_type = param_dict['Name']['sim type']


# =====================
# Construct Time Vector
# =====================
t0,tn,dt = param_dict['Time'].values()
t = np.arange(t0,tn,dt)

# ===============================
# Construct Chaotic Input Current
# ===============================
data = np.load(f'{args.data_file}',allow_pickle=True)[()]
I_inj = data['Validation']['I_inj']

# =================================
# Get Noise Current Hyperparameters
# =================================
snr,tau_noise,g_syn_max,spiker_fr = param_dict['Noise Current'].values()


# ====================================
# Simulate with different noise inputs
# ====================================
spikes = []

for trial in range(args.n_trials):
    print(f'Trial: {trial}')
    I_noise = balanced_noise_current(spiker_fr, dt, t, g_syn_max, tau_noise)
    # Scale noise to have fixed SNR to I_inj
    I_noise = scale_noise_to_SNR(I_inj, I_noise, snr)
    V = stimulate(t, I_inj, I_noise, args.config_file, monitor=True)
    spikes.append(rasterize(t[::args.DSR],V[::args.DSR]))
spikes_dict = {'spike times':spikes,'n trials': args.n_trials,'Time': [t0,tn,dt]}
np.save(f'{args.out_path}/null_spikes_{sim_type}.npy',spikes_dict)




