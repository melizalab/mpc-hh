import sys
sys.path.append('neuron_scripts')
sys.path.append('DDF_scripts')
import argparse
import numpy as np
from connor_stevens import config_dict
from waveform_analysis import *
from stimulate_neuron import *

'''
Uses the CS model to predict membrane voltage when no I_noise is present. This can be used to evaluate how
good the DDF model was. If the DDF model was perfect, the best it could do is recreate this voltage trace since
it has no knowledge of the I_noise current.
'''

# ============================
# Parse Command Line Arguments
# ============================
p = argparse.ArgumentParser()
p.add_argument('--data_file', default='assimilation_data/type_I.npy', type=str, help='file path to assimilation data')
p.add_argument('--config_file', default='config/config_cs_type_I.yaml', type=str, help='file path to config file for assimilation data')
p.add_argument('--out_path', default = 'open_loop_scripts', type = str, help = 'path to output data')
p.add_argument('--DSR', default=5, type=int, help='downsample rate of model (i.e. 5 -> only use every 5 measurements')
args = p.parse_args()

# ===========
# Import Data
# ===========
dsr = args.DSR
param_dict = config_dict(args.config_file)
data = np.load(f'{args.data_file}',allow_pickle=True)[()]
I_inj = data['Validation']['I_inj']
sim_type = param_dict['Name']['sim type']

# =========
# Stimulate
# =========
t0,tn,dt = param_dict['Time'].values()
t = np.arange(t0,tn,dt)
V = stimulate(t,I_clean = I_inj,I_noise = np.zeros_like(I_inj),config_file=args.config_file,monitor=True)
spikes = rasterize(t,V)
perfect_model = {'V':V,'spike':spikes}
# Change path as needed
np.save(f'open_loop_scripts/type_{sim_type}_ref.npy',perfect_model)
