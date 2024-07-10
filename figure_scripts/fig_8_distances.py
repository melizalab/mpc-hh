import numpy as np
import pyspike as spk
import sys
sys.path.append('neuron_scripts')
from waveform_analysis import rasterize

'''
Creates the ISI- and Spike-Distances for Figure 8
To rerun code, you will need to have a path to the control outputs.
'''

sim_type = 'type_I'
trial = 0 # trial index to use
# "Continuous" and downsampled discrete time arrays
t = np.arange(0,500,.02)
T = np.arange(0,500,.1)
# MPC Data
mpc_data = np.load(f'control_output/mpc_performance_{sim_type}_exp_3.npy',allow_pickle=True)[()]
print('MPC Data Type I')
print(f'----> ISI for trial {trial}:',np.round(mpc_data['ISIs'][trial],2))
print(f'----> Spike for trial {trial}:',np.round(mpc_data['SpkDs'][trial],2))
# Pulse
pulse_data = np.load(f'open_loop_output/open_loop_performance_{sim_type}_exp_3.npy',allow_pickle=True)[()]
print('Pulse Data')
print(f'----> ISI for trial {trial}:',np.round(pulse_data['ISIs'][trial],2))
print(f'----> Spike for trial {trial}:',np.round(pulse_data['SpkDs'][trial],2))

# P-Controller
p_data= np.load(f'p_control_output/exp_3/type_I_trial_indx_{trial}.npy',allow_pickle=True)[()]
ref_spikes = rasterize(T,p_data['reference trajectory'])
control_spikes = rasterize(t,p_data['V_actual'])
ref_spike_train = spk.SpikeTrain(ref_spikes, [0, 500])
control_spike_train = spk.SpikeTrain(control_spikes, [0, 500])
p_ISI = spk.isi_distance(ref_spike_train, control_spike_train, interval=(0, 500))
p_spike = spk.spike_distance(ref_spike_train, control_spike_train, interval=(0, 500))
print('P-controller Data')
print(f'----> ISI for trial {trial}:',np.round(p_ISI,2))
print(f'----> Spike for trial {trial}:',np.round(p_spike,2))


sim_type = 'type_II'
trial = 32 #trial indx to use
# MPC Data
mpc_data = np.load(f'control_output/mpc_performance_{sim_type}_exp_3.npy',allow_pickle=True)[()]
print('MPC Data Type II')
print(f'----> ISI for trial {trial}:',np.round(mpc_data['ISIs'][trial],2))
print(f'----> Spike for trial {trial}:',np.round(mpc_data['SpkDs'][trial],2))
# Pulse
pulse_data = np.load(f'open_loop_output/open_loop_performance_{sim_type}_exp_3.npy',allow_pickle=True)[()]
print('Pulse Data')
print(f'----> ISI for trial {trial}:',np.round(pulse_data['ISIs'][trial],2))
print(f'----> Spike for trial {trial}:',np.round(pulse_data['SpkDs'][trial],2))


# P-Controller
p_data= np.load(f'p_control_output/exp_3/type_II_trial_indx_{trial}.npy',allow_pickle=True)[()]
ref_spikes = rasterize(T,p_data['reference trajectory'])
control_spikes = rasterize(t,p_data['V_actual'])
ref_spike_train = spk.SpikeTrain(ref_spikes, [0, 500])
control_spike_train = spk.SpikeTrain(control_spikes, [0, 500])
p_ISI = spk.isi_distance(ref_spike_train, control_spike_train, interval=(0, 500))
p_spike = spk.spike_distance(ref_spike_train, control_spike_train, interval=(0, 500))
print('P-controller Data')
print(f'----> ISI for trial {trial}:',np.round(p_ISI,2))
print(f'----> Spike for trial {trial}:',np.round(p_spike,2))
breakpoint()