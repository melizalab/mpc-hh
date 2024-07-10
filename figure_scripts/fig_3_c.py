import sys
sys.path.append('neuron_scripts')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from connor_stevens import *
from neuron_inputs import *
from waveform_analysis import rasterize

'''
Creates the plots for Figure 3c
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


# ===============================
# Random seed for reproducibility
# ===============================
n_trials = 10
random_seeds = np.arange(n_trials)

# ==================================
# Time Parameters and Injected Input
# ==================================
# 500 ms pulse
t0 = 0
tn = 500
dt = 0.02
t = np.arange(t0,tn,dt)
I_inj = np.zeros_like(t)
'''
Modify I_inj to be a 300 ms pulse with amplitude 9
starting at 100ms and ending at 400ms.
'''
pulse_start_time = 100
pulse_end_time = 400
I_inj[int(pulse_start_time/dt):int(pulse_end_time/dt)] = 9

# ===============================================
# Get Noise Input (parameters for Poisson spikes)
# ===============================================
g_syn_max = 5
tau_syn = 3
spiker_fr = 20
snr = 5

# ======================
# Simulate Type I Neuron
# ======================
config_file = 'config/config_cs_type_I.yaml'
type_I_neuron = construct_neuron(config_file)
for trial in range(n_trials):
    print(f'trial:{trial}')
    np.random.seed(random_seeds[trial])
    # Initial conditions
    X0 = np.concatenate((-72.8 + np.random.normal(0, 1, 1), np.random.uniform(0, 1, 5)))
    V = np.zeros_like(t)
    V[0] = X0[0]
    # Get noise current and scale to appropriate SNR
    I_noise = balanced_noise_current(spiker_fr, dt, t, g_syn_max, tau_syn)
    I_noise = scale_noise_to_SNR(I_inj, I_noise, snr)
    # Simulate
    for i in range(1, len(t)):
        tspan = [t[i - 1], t[i]]
        X = odeint(type_I_neuron.ode_eqs, X0, tspan, args=(I_inj[i]+I_noise[i],))
        X0 = X[1]
        V[i] = X0[0]
    spike_times = rasterize(t, V)
    plt.vlines(spike_times, 0 + trial, 0.8 + trial,color='darkcyan')
plt.xlim([t0,tn])
plt.xticks([0,250,500])
#plt.savefig('figure_scripts/fig_3_c_type_I.pdf')
plt.show()


# ======================
# Simulate Type I Neuron
# ======================
config_file = 'config/config_cs_type_II.yaml'
type_II_neuron = construct_neuron(config_file)
for trial in range(n_trials):
    print(f'trial:{trial}')
    np.random.seed(random_seeds[trial])
    # Initial conditions
    X0 = np.concatenate((-72.8 + np.random.normal(0, 1, 1), np.random.uniform(0, 1, 5)))
    V = np.zeros_like(t)
    V[0] = X0[0]
    # Get noise current and scale to appropriate SNR
    I_noise = balanced_noise_current(spiker_fr, dt, t, g_syn_max, tau_syn)
    I_noise = scale_noise_to_SNR(I_inj, I_noise, snr)
    # Simulate
    for i in range(1, len(t)):
        tspan = [t[i - 1], t[i]]
        X = odeint(type_II_neuron.ode_eqs, X0, tspan, args=(I_inj[i]+I_noise[i],))
        X0 = X[1]
        V[i] = X0[0]
    spike_times = rasterize(t, V)
    plt.vlines(spike_times, 0 + trial, 0.8 + trial,color='darkmagenta')
plt.xlim([t0,tn])
plt.xticks([0,250,500])
#plt.savefig('figure_scripts/fig_3_c_type_II.pdf')
plt.show()
