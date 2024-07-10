import sys
sys.path.append('neuron_scripts')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from connor_stevens import *
from neuron_inputs import *

'''
Creates the plots for Figure 3 a
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
np.random.seed(10)

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

# ======================
# Simulate Type I Neuron
# ======================
config_file = 'config/config_cs_type_I.yaml'
type_I_neuron = construct_neuron(config_file)
# Initial Conditions
X0 = np.concatenate((-72.8+np.random.normal(0,1,1), np.random.uniform(0, 1, 5)))
V = np.zeros_like(t)
V[0] = X0[0]
# Simulate
for i in range(1, len(t)):
    tspan = [t[i - 1], t[i]]
    X = odeint(type_I_neuron.ode_eqs, X0, tspan, args=(I_inj[i],))
    X0 = X[1]
    V[i] = X0[0]
plt.plot(t,V,color='darkcyan')
plt.yticks([-50,0,50])
plt.xticks([0,250,500])
plt.ylim([-80,60])
#plt.savefig('figure_scripts/fig_3_a_type_I.pdf')
plt.show()

# ======================
# Simulate Type II Neuron
# ======================
config_file = 'config/config_cs_type_II.yaml'
type_II_neuron = construct_neuron(config_file)
# Initial Conditions
X0 = np.concatenate((-72.8+np.random.normal(0,1,1), np.random.uniform(0, 1, 5)))
V = np.zeros_like(t)
V[0] = X0[0]
# Simulate
for i in range(1, len(t)):
    tspan = [t[i - 1], t[i]]
    X = odeint(type_II_neuron.ode_eqs, X0, tspan, args=(I_inj[i],))
    X0 = X[1]
    V[i] = X0[0]
plt.plot(t,V,color='darkmagenta')
plt.yticks([-50,0,50])
plt.xticks([0,250,500])
plt.ylim([-80,60])
#plt.savefig('figure_scripts/fig_3_a_type_II.pdf')
plt.show()