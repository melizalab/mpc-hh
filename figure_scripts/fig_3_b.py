import sys
sys.path.append('neuron_scripts')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from connor_stevens import *
from neuron_inputs import *
from waveform_analysis import rasterize

'''
Creates the plots used for Figure 3b
'''

# ===============
# Plotting Params
# ===============
fontsize = 16
tick_thickness = 1
tick_length = 8
plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize,
                     'xtick.major.width': tick_thickness, 'ytick.major.width': tick_thickness,
                     'xtick.major.size': tick_length, 'ytick.major.size': tick_length})

# ===============================
# Random seed for reproducibility
# ===============================
np.random.seed(0)

# =========================================
# Time Parameters and Injected Input Values
# =========================================
# Repeat a 2s step current stimulus 30 times with values between 6 and 10
t0 = 0
tn = 2000
dt = 0.02
t = np.arange(t0,tn,dt)
n_inputs = 30
I_inj_values = np.linspace(6,10,n_inputs)

# =============
# Type I Neuron
# =============
type_I_FR = np.zeros(len(I_inj_values))
for i in range(n_inputs):
    print(f'Input:{i}')
    config_file = 'config/config_cs_type_I.yaml'
    type_I_neuron = construct_neuron(config_file)
    # Initial conditions
    X0 = np.concatenate((-72.8 + np.random.normal(0, 1, 1), np.random.uniform(0, 1, 5)))
    V = odeint(type_I_neuron.ode_eqs, X0, t, args=(I_inj_values[i],))[:,0]
    # buffer: number of time steps to remove from firing rate calculation since initial conditions can produce a spontaneous spike
    buffer = 500
    spikes = rasterize(t[buffer:],V[buffer:])
    type_I_FR[i] = len(spikes)/(((tn-buffer*dt)-t0)/1000)
plt.plot(I_inj_values,type_I_FR,color='darkcyan')
plt.yticks([0,10,20])
#plt.savefig('fig_3_b_type_I.pdf')
plt.show()


# =============
# Type I Neuron
# =============
type_II_FR = np.zeros(len(I_inj_values))
for i in range(n_inputs):
    print(f'Input:{i}')
    config_file = 'config/config_cs_type_II.yaml'
    type_II_neuron = construct_neuron(config_file)
    # Initial conditions
    X0 = np.concatenate((-72.8 + np.random.normal(0, 1, 1), np.random.uniform(0, 1, 5)))
    V = odeint(type_II_neuron.ode_eqs, X0, t, args=(I_inj_values[i],))[:,0]
    # buffer: number of time steps to remove from firing rate calculation since initial conditions can produce a spontaneous spike
    buffer = 500
    spikes = rasterize(t[buffer:],V[buffer:])
    type_II_FR[i] = len(spikes)/(((tn-buffer*dt)-t0)/1000)
plt.plot(I_inj_values[:9],type_II_FR[:9],color='darkmagenta')
plt.plot(I_inj_values[9:],type_II_FR[9:],color='darkmagenta')
plt.yticks([0,50,100])
#plt.savefig('fig_3_b_type_II.pdf')
plt.show()


