import numpy as np
import yaml
'''
Purpose:
This script is for generating responses of connor-stevens type neurons.
It contains functions for:
    + conductance-based model (cs_neuron)
    + gating variable kinetics (e.g. tau_a, a_inf, b_h)
'''


class Neuron:
    def __init__(self, sim_type):
        # Simulation type (ie 'I' or 'II)
        self.sim_type = sim_type
        # Membrane capacitance (\mu F cm^-2)
        self.C_m = 1
        # Reversal Potentials (mV)
        self.E_na = 50
        self.E_k = -77
        self.E_a = -80
        # Maximal conductances (mS cm^-2)
        self.g_na = 120
        self.g_k = 20
        self.g_a = 47.7
        self.g_l = 0.3
        self.I_e = 0
        # The leak current reversal potential is different for type 'I' and type 'II' neurons
        if self.sim_type == 'I':
            self.E_l = -22
        elif self.sim_type == 'II':
            self.E_l = -72.8

    # Kinetic equations
    def a_inf(self, V):
        numer = 0.0761 * np.exp((V + 99.22) / 31.84)
        denom = 1 + np.exp((V + 6.17) / 28.93)
        return (numer / denom) ** (1 / 3)

    def b_inf(self, V):
        numer = 1
        denom = (1 + np.exp((V + 58.3) / 14.54)) ** 4
        return numer / denom

    def tau_a(self, V):
        numer = 1.158
        denom = 1 + np.exp((V + 60.96) / 20.12)
        return 0.3632 + (numer / denom)

    def tau_b(self, V):
        numer = 2.678
        denom = 1 + np.exp((V - 55) / 16.027)
        return 1.24 + (numer / denom)

    def a_m(self, V):
        numer = -0.1 * (V + 34.7)
        denom = np.exp(-(V + 34.7) / 10) - 1
        return 3.8 * numer / denom

    def a_h(self, V):
        return 3.8 * 0.07 * np.exp(-(V + 53) / 20)

    def a_n(self, V):
        numer = -0.01 * (V + 50.7)
        denom = np.exp(-(V + 50.7) / 10) - 1
        return (3.8 / 2) * (numer / denom)

    def b_m(self, V):
        return 3.8 * 4 * np.exp(-(V + 59.7) / 18)

    def b_h(self, V):
        numer = 3.8
        denom = np.exp(-(V + 23) / 10) + 1
        return numer / denom

    def b_n(self, V):
        return (3.8 / 2) * 0.125 * np.exp(-(V + 60.7) / 80)

    # Connor-Stevens dynamic equations
    def eqs(self, V, a, b, m, h, n, I):
        # Call this function instead of ode_eqs if using do-mpc simulator object
        if self.sim_type == 'I':
            I_a = self.g_a * (a ** 3) * b * (V - self.E_a)
        elif self.sim_type == 'II':
            I_a = 0
        # Intrinsic currents
        self.I_na = self.g_na * (m ** 3) * h * (V - self.E_na)
        I_k = self.g_k * (n ** 4) * (V - self.E_k)
        I_l = self.g_l * (V - self.E_l)
        # ODEs for state variables
        dVdt = (1 / self.C_m) * (I + self.I_e - self.I_na - I_k - I_a - I_l)
        dadt = (self.a_inf(V) - a) / self.tau_a(V)
        dbdt = (self.b_inf(V) - b) / self.tau_b(V)
        dmdt = self.a_m(V) * (1 - m) - self.b_m(V) * m
        dhdt = self.a_h(V) * (1 - h) - self.b_h(V) * h
        dndt = self.a_n(V) * (1 - n) - self.b_n(V) * n
        return dVdt, dadt, dbdt, dmdt, dhdt, dndt

    # Call when using scipy.integrate.odeint to simulate responses
    def ode_eqs(self, state, t, I):
        V, a, b, m, h, n = state
        return self.eqs(V, a, b, m, h, n, I)


def config_dict(config_file):
    # Useful function to take file path of config file and load it to a variable
    with open(config_file,'r') as file:
        config = yaml.safe_load(file)
    return config

def construct_neuron(config_file):
    # Creates an instance of a CS neuron using path to config file
    config = config_dict(config_file)
    neuron = Neuron(sim_type=config['Name']['sim type'])
    return neuron