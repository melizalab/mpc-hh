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
p.add_argument('--out_path', default = 'assimilation_data', type = str, help = 'file path to save produced assimilation data.')
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
input_scaling,tau_inj = param_dict['Injected Current'].values()
I_inj_train = chaotic_current(t0,tn,dt,input_scaling,tau_inj,random_seed=0) # Training
I_inj_val = chaotic_current(t0,tn,dt,input_scaling,tau_inj,random_seed=1) # Validation

# ================================================
# Construct Noise Current For Training and Testing
# ================================================
snr,tau_noise,g_syn_max,spiker_fr = param_dict['Noise Current'].values()
I_noise_train = balanced_noise_current(spiker_fr, dt, t, g_syn_max, tau_noise)
I_noise_train = scale_noise_to_SNR(I_inj_train, I_noise_train, snr) # Training
I_noise_val = balanced_noise_current(spiker_fr, dt, t, g_syn_max, tau_noise)
I_noise_val = scale_noise_to_SNR(I_inj_val, I_noise_val, snr) # Validation

# ================
# Stimulate Neuron
# ================
print('Constructing Training Data...')
V_train = stimulate(t,I_inj_train,I_noise_train,args.config_file,monitor=True)
print('Constructing Validation Data...')
V_val = stimulate(t,I_inj_val,I_noise_val,args.config_file,monitor=True)

# =========
# Save Data
# =========
data = {'Train':{
            'V':V_train,
            'I_inj':I_inj_train,
            'I_noise':I_noise_train},
        'Validation':{
            'V':V_val,
            'I_inj':I_inj_val,
            'I_noise':I_noise_val}
        }
file_path = f'{args.out_path}/type_{sim_type}'
np.save(file_path,data)
