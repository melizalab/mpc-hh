import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('neuron_scripts')
from waveform_analysis import rasterize
import pyspike as spk

'''
Evaluates MPC performance and also can generate Figure 6a
'''

# Number of reference trajectories to check
n_trials = 50
# What kind of CS model to use (e.g. 'I' or 'II')
sim_type = 'I'
# Which experiment to evaluate
exp_num = 'exp_1'

# Collect fit metrics
MSEs = np.zeros(n_trials)
ISIs = np.zeros(n_trials)
SpkDs = np.zeros(n_trials)

# Plot a specific trial, used to make Figure 6a
inspect_trial = 0

for i in range(n_trials):
    # Load data
    data = np.load(f'control_output/{exp_num}/type_{sim_type}_trial_indx_{i}.npy',allow_pickle=True)[()]

    # Get reference and controlled trajectories
    ref_traj = data['reference trajectory']
    V_control = data['V_control']

    # Get time metadata
    dt,t0,tn,delta_t = data['Time'].values()
    t = np.arange(t0,tn,dt)
    T = np.arange(t0,tn,delta_t)

    # Convert trajectories into spike trains
    ref_spikes = rasterize(T,ref_traj[:len(T)])
    control_spikes = rasterize(T,V_control[:len(T)])
    ref_spike_train = spk.SpikeTrain(ref_spikes, [t0, tn])
    control_spike_train = spk.SpikeTrain(control_spikes, [t0, tn])

    # Get fit metrics
    MSEs[i] = mean_squared_error(ref_traj,V_control)
    ISIs[i] = spk.isi_distance(ref_spike_train, control_spike_train, interval=(t0, tn))
    SpkDs[i] = spk.spike_distance(ref_spike_train, control_spike_train, interval=(t0, tn))

    if i == inspect_trial:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].vlines(ref_spikes, 60, 65, color='black')
        ax[0].vlines(control_spikes, 70, 75, color='red')
        ax[0].plot(T, ref_traj[:len(T)], color='black', alpha=0.7)
        ax[0].plot(T, V_control[:len(T)], color='red', alpha=0.7)
        ax[1].plot(t, data['I_control'], color='black', alpha=0.7)
        ax[1].plot(t, data['I_noise'], color='darkmagenta', alpha=0.7)
        #plt.savefig(f'figure_parts/figure_6_a_type_{sim_type}.pdf')
        plt.show()


print(f'MSE: mean = {np.mean(MSEs):.2f}, SD = {np.std(MSEs):.2f}')
print(f'ISI Distance: mean = {np.mean(ISIs):.2f}, SD = {np.std(ISIs):.2f}')
print(f'Spike Distance: mean = {np.mean(SpkDs):.2f}, SD = {np.std(SpkDs):.2f}')
out_file = f'control_output/mpc_performance_type_{sim_type}_{exp_num}.npy'
data_file = {'MSEs':MSEs,'ISIs':ISIs,'SpkDs':SpkDs}
np.save(out_file,data_file)
