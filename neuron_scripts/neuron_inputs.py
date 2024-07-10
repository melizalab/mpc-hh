import sys
sys.path.append('neuron_scripts')
import numpy as np
from poisson_spiking import *
from scipy.integrate import odeint

def balanced_noise_current(firing_rate,dt,t,g_syn_max,tau):
    '''
    Creates post-synaptic potential time series of balanced excitation and inhibition
    :param firing_rate: Firing rate in Hz
    :param dt: time step of simulation given in ms (ie 0.1 = 0.1 ms)
    :param t: vector of time values for simulation
    :param g_syn_max: maximum conductance of the PSP
    :param tau: time decay parameter
    :return: a PSP with balanced excitatory and inhibitory currents
    '''
    # Both excitatory and inhibitory neurons have same spike rate
    spiker = rate_neuron(dt)
    # Get excitatory and inhibitory spike trains
    exc_spike_train = np.array([spiker.generate_spike(rate=firing_rate) for _ in t])
    inh_spike_train = np.array([spiker.generate_spike(rate=firing_rate) for _ in t])
    # Convert to spike times
    exc_spike_times = t[np.where(exc_spike_train > 0)]
    inh_spike_times = t[np.where(inh_spike_train > 0)]
    # Convert to post-synaptic potentials
    exc_PSP = spikes_to_PSP(exc_spike_times, t, dt, g_syn_max, tau)
    inh_PSP = spikes_to_PSP(inh_spike_times, t, dt, -g_syn_max, tau)
    # Sum for total noise input current
    I_noise = exc_PSP + inh_PSP
    return I_noise


def lorenz63(initial_conditions,t,tau):
    '''
    Simulates the Lorenz 63 system
    :param initial_conditions: 3-dimensional array of initial conditions
    :param t: time arg for odeint
    :param tau: time scaling constant
    :return: derivatives of Lorenz 63 system
    '''
    x,y,z = initial_conditions
    sigma,beta,rho = [10.0, 8.0/3.0, 28.0]
    dxdt = sigma*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y-beta*z
    return np.array([dxdt,dydt,dzdt])*tau

def chaotic_current(t0,tn,dt,input_scaling = 1, tau = .05, initial_conditions = None, remove_transient = True, random_seed = None):
    '''
    Generates input current for the connor-stevens model based on the lorenz 63 system
    :param t0: initial time (ms)
    :param tn: final time (ms)
    :param dt: time bin size
    :param input_scaling: how much to scale the input current by
    :param initial_conditions: initial conditions of lorenz 63 system
    :param tau: time scaling for input current
    :param remove_transient: boolean to remove transient response of lorenz 63 system (heuristically coded)
    :param random_seed: random seed to ensure reproducibility (only for input stimulus)
    :return: input current from lorenz system
    '''
    # Set random seed if applicable
    if random_seed is not None: np.random.seed(random_seed)

    # checks boolean to remove transient response if desired
    buffer = 0 if remove_transient == False else int(100/dt/tau)

    # create time vector
    x = int((tn-t0)/dt)+buffer
    t = np.linspace(t0,tn,x)

    # get initial conditions of lorenz 63 system
    if initial_conditions is None: initial_conditions = np.random.normal(0,1,3)

    # create clean input current
    X = odeint(lorenz63,initial_conditions,t, args = (tau,))
    I = input_scaling*X[:,0]

    # remove number of buffer samples for when removing transient response of lorenz 63
    I = I[buffer:]

    return I

def SNR(signal,noise):
    '''
    Calculates the signal to noise ratio (SNR)
    :param signal: duh
    :param noise: duh
    :return: the SNR
    '''
    return np.mean(signal**2)/np.mean(noise**2)

def scale_noise_to_SNR(signal,noise,SNR):
    '''
    Scale noise to have desired SNR (assumes signal and noise have same length!!!)
    :param signal: duh
    :param noise: duh (thing to scale)
    :param SNR: desired SNR
    :return: scaled noise that now has desired SNR with signal
    '''
    a = np.sqrt(np.sum(signal**2)/(SNR*np.sum(noise**2)))
    return a*noise