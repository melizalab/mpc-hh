def extract_waveform(t,V,threshold = 30,buffer = 40):
    '''
    Extracts the average waveform for an array of membrane voltages
    :param t: vector of time values
    :param V: vector of membrane voltages
    :param threshold: value of membrane voltage to determine if there is a spike
    :param buffer: how many time samples before and after a spike to extract
    :return:
    '''
    spike = False
    spk_t = []
    waveforms = []
    for indx,v in enumerate(V):
        if v >= threshold and spike == False:
            spk_t.append(t[indx])
            wf = V[indx-buffer:indx+buffer]
            if len(wf) > 0: waveforms.append(wf)
            spike = True
        else:
            spike = False
    return spk_t,waveforms

def rasterize(t,V,threshold = 30):
    '''
    Takes an array of membrane voltages and returns a spike train
    :param t: vector of time values
    :param V: vector of membrane voltages
    :param threshold: value fo membrane voltage to determine if there is a spike
    :return: an array of spike times
    '''
    spike = False
    spk_t = []
    for indx,v in enumerate(V):
        if v>= threshold and spike == False:
            spk_t.append(t[indx])
            spike = True
        elif v< threshold:
            spike = False
    return spk_t
