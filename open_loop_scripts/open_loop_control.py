import sys
sys.path.append('neuron_scripts')
from waveform_analysis import rasterize
from connor_stevens import *
from stimulate_neuron import *
import argparse
from neuron_inputs import *
import numpy as np
from scipy.integrate import odeint
from neuron_inputs import *

'''
Open-loop control for experiments 1,2, and 3. 
For exp_num=1 or exp_num=2, a previously generated injected current is used.
For exp_num=3, a pulse current is used.
'''

# =======================
# Parse Command Line Args
# =======================
p = argparse.ArgumentParser()
p.add_argument('--config_file', default='config/config_cs_type_I.yaml', type=str, help='file path to config file for assimilation data')
p.add_argument('--ddf_path', default = 'saved_models/ddf_type_I.npy', type = str, help = 'path to DDF model.')
p.add_argument('--out_path',default = 'open_loop_output', type = str, help = 'path to where output is to be saved.')
p.add_argument('--exp_num',default=3,type=int)
p.add_argument('--trial_indx',default =0, type = int, help = 'which trial to use in reference data.')
p.add_argument('--pulse_width',default=1,type=int, help = '1/2 width of pulse current (in ms), only needed for exp_num=3')
p.add_argument('--pulse_amplitude',default = 50.0, type = float, help='amplitude of pulse current, only needed for exp_num=3')
args = p.parse_args()


# ================
# Construct Neuron
# ================
neuron = construct_neuron(args.config_file)
param_dict = config_dict(args.config_file)
sim_type = param_dict['Name']['sim type']

# =============
# Get Real Time
# =============
t0,tn,dt = param_dict['Time'].values()
t = np.arange(t0,tn,dt)

# ===========================================
# Load Reference Trajectory and Noise Current
# ===========================================
if args.exp_num == 1:
    ref_data = np.load(f'reference_trajectories/type_{sim_type}_ref_trajectories.npy',allow_pickle=True)[()]
    I_inj = ref_data['I_inj'][args.trial_indx][:len(t)]

elif args.exp_num == 2:
    if sim_type == 'I':
        ref_type = 'II'
    else:
        ref_type = 'I'
    ref_data = np.load(f'reference_trajectories/type_{ref_type}_ref_trajectories.npy',allow_pickle=True)[()]
    I_inj = ref_data['I_inj'][args.trial_indx][:len(t)]

elif args.exp_num == 3:
    ref_data = np.load(f'reference_trajectories/type_{sim_type}_waveform_ref_trajectories.npy',allow_pickle=True)[()]


# Get Noise Current From MPC Trials
mpc_data = np.load(f'control_output/exp_{args.exp_num}/type_{sim_type}_trial_indx_{args.trial_indx}.npy',allow_pickle=True)[()]
I_noise = mpc_data['I_noise'][:len(t)]


# ========================
# Get Reference Trajectory
# ========================
ref_traj = ref_data['V'][args.trial_indx][:len(t)]

# ================
# Construct Neuron
# ================
neuron = construct_neuron(args.config_file)

# ==============================
# Stimulate with Open-Loop I_inj
# ==============================
if args.exp_num != 3:
    # Open-loop in experiments 1 and 2 where repeats of the original I_inj
    V_actual = stimulate(t,I_inj,I_noise,args.config_file,monitor=False)
else:
    # Open-loop in experiment 3 was a pulse current
    # Create instance of connor-stevens neuron
    neuron = construct_neuron(args.config_file)

    # Initialize neuron
    V_actual = np.zeros_like(t)
    X0 = np.random.uniform(0, 1, 6)
    V_actual[0] = -72.8+np.random.normal(0,1,1)
    X0[0] = V_actual[0]

    # Get spikes times
    spk_ts = rasterize(t, ref_traj)
    pulse_times = np.array([(time - args.pulse_width, time + args.pulse_width) for time in spk_ts])
    pulse_indx = 0
    I_inj = np.zeros_like(t)
    I_inj[0] = 0
    pulse = False
    # Simulate response
    for i in range(1,len(t)):
        tspan = [t[i-1], t[i]]
        # Pulse
        if t[i-1] > pulse_times[pulse_indx][0] and t[i-1] < pulse_times[pulse_indx][1]:
            U = args.pulse_amplitude
            pulse = True
        else:
            if pulse == True:
                if pulse_indx != len(spk_ts)-1:
                    pulse_indx+=1
                pulse=False
            U = 0

        I_inj[i] = U
        X = odeint(neuron.ode_eqs, X0, tspan, args=(I_inj[i]+I_noise[i],))
        X0 = X[1]
        V_actual[i] = X0[0]

# =========
# Save Data
# =========
out_file = f'{args.out_path}/exp_{args.exp_num}/type_{sim_type}_trial_indx_{args.trial_indx}.npy'
print(f'Saving data: {out_file}')
data_dict = {
    'reference trajectory':ref_traj,
    'V_actual': V_actual,
    'I_control': I_inj,
    'I_noise': I_noise,
    'Time': {
        'dt': dt,
        't0': t0,
        'tn': tn,
    }
}
np.save(out_file, data_dict)
