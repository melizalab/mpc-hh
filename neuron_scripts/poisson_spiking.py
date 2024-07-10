import numpy as np

class rate_neuron():
    # Constructs a Poisson neuron with a static firing rate
    def __init__(self,dt):
        '''
        :param dt: time step of simulation given in ms (eg 0.1 == 0.1 ms)
        '''
        self.dt = dt
    def generate_spike(self,rate):
        '''
        Probability of spiking at a time bin is given by p. The rate is divided by 1000 since dt is
        in units of ms and rate is in units of Hz.
        :param rate: firing rate of the neuron in Hz (eg 10 == 10Hz).
        :return: Returns 0 if no spike, returns 1 if a spike. If the probability p is greater than 1,
        it is clipped to be 1.
        '''
        rate /= 1000
        p = rate*self.dt
        if p >1:
            return 1
        else:
            return np.random.binomial(1,p)

def spikes_to_PSP(spike_times,t,dt,g_syn_max,tau):
    '''
    Converts an array of spike times into a continuous post-synaptic potential (PSP) where
    an individual PSP is an alpha function beginning at the single spike time.
    :param spike_times: the array of spikes times
    :param t: time array
    :param dt: time step in time array
    :param g_syn_max: max synaptic conductance
    :param tau: decay time parameter
    :return: total PSP produced by spikes (sum of the individual PSPs)
    '''
    PSP = np.zeros_like(t)
    for spk_t in spike_times:
        t_indx = int(spk_t/dt)
        spike_diff = (1/tau)*(t[t_indx:]-spk_t)
        PSP[t_indx:]+= g_syn_max*spike_diff*np.exp(-spike_diff)
    return PSP