import numpy as np
from scipy.integrate import odeint
# Import custom modules
from connor_stevens import *

def stimulate(t,I_inj,I_noise,config_file,random_seed = None,monitor = False):
    '''
    Simulates the responses of a CS neuron being stimulated by and I_inj and I_noise current
    :param t: vector of time values used in simulation
    :param I_inj: known injected stimulating current
    :param I_noise: unknown noise current
    :param config_file: path to config file that has CS neuron parameters
    :param random_seed: random seed for the initial conditions to be reproducible
    :param monitor: if True, print message every 1000 dt iterations for debugging
    :return: membrane voltage produced from I_inj+I_noise
    '''

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create instance of connor-stevens neuron
    neuron = construct_neuron(config_file)

    # Initialize neuron
    V = np.zeros_like(t)
    X0 = np.random.uniform(0, 1, 6)
    V[0] = -72.8+np.random.normal(0,1,1)
    X0[0] = V[0]

    # Total Input
    I = I_inj+I_noise

    # Simulate response
    for i in range(1,len(t)):
        if monitor == True:
            if i%1000 == 0:
                print(f'Simulation iteration: {i}/{len(t)}')
        tspan = [t[i-1], t[i]]
        X = odeint(neuron.ode_eqs, X0, tspan, args=(I[i],))
        X0 = X[1]
        V[i] = X0[0]
    return V

def gap_junction_stimulate(t,I_n1,I_n2,g1,g2,n1_config_file,n2_config_file,monitor=False):
    '''
    UNUSED IN MANUSCRIPT: couples two CS neurons with gap junction
    :param t: vector of time values used in simulation
    :param I_n1: injected current into neuron 1
    :param I_n2: injected current into neuron 2
    :param g1: gap junction conductance for neuron 1
    :param g2: gap junction conductance for neuron 2
    :param n1_config_file: config file of CS model parameters for neuron 1
    :param n2_config_file: config file of CS model parameters for neuron 2
    :param monitor: if True, print message every 1000 dt iterations for debugging
    :return: membrane voltages for the two neurons
    '''
    # Construct Neurons
    param_dict_n1 = config_dict(n1_config_file)
    param_dict_n2 = config_dict(n2_config_file)
    neuron_1 = construct_neuron(n1_config_file)
    neuron_2 = construct_neuron(n2_config_file)

    # Init Neurons
    V_n1 = np.zeros_like(t)
    X0_n1 = np.random.uniform(0, 1, 6)
    V_n1[0] = -72.8 + np.random.normal(0, 1, 1)
    X0_n1[0] = V_n1[0]

    V_n2 = np.zeros_like(t)
    X0_n2 = np.random.uniform(0, 1, 6)
    V_n2[0] = -72.8 + np.random.normal(0, 1, 1)
    X0_n2[0] = V_n2[0]

    for i in range(1, len(t)):
        # Monitor Progress
        if monitor ==True:
            if i % 1000 == 0:
                print(f'Iteration: {i}/{len(t)}')
        # Time
        tspan = [t[i - 1], t[i]]
        # Neuron 1
        X_n1 = odeint(neuron_1.ode_eqs, X0_n1, tspan, args=(I_n1[i] + g1 * (V_n2[i - 1] - V_n1[i - 1]),))
        X0_n1 = X_n1[1]
        V_n1[i] = X0_n1[0]
        # Neuron 2
        X_n2 = odeint(neuron_2.ode_eqs, X0_n2, tspan, args=(I_n2[i] + g2 * (V_n1[i - 1] - V_n2[i - 1]),))
        X0_n2 = X_n2[1]
        V_n2[i] = X0_n2[0]
    return V_n1,V_n2