import sys
sys.path.append('neuron_scripts')
sys.path.append('DDF_scripts')
from DDF import *
import numpy as np
import argparse
import matplotlib.pyplot as plt
from connor_stevens import config_dict
from waveform_analysis import *
from sklearn.linear_model import RidgeCV
import pyspike as spk

# ============================
# Parse Command Line Arguments
# ============================
p = argparse.ArgumentParser()
p.add_argument('--data_file', default='assimilation_data/type_I.npy', type=str, help='file path to assimilation data')
p.add_argument('--config_file', default='config/config_cs_type_I.yaml', type=str, help='file path to config file for assimilation data')
p.add_argument('--out_path', default = 'saved_models/ddf_type_I', type=str, help = 'path to output of DDF model.')
p.add_argument('--D_e', default = 2, type = int, help = 'delay embedding dimension.')
p.add_argument('--tau', default = 1, type = int, help = 'time steps skipped in delay embedding.')
p.add_argument('--n_centers', default = 50, type = int, help = 'number of centers in RBF network.')
p.add_argument('--scale', default = .01, type = float, help = 'scaling parameter for each RBF in network.')
p.add_argument('--clustering_method', default='kmeans',type = str, help = 'how to find centers (ie. kmeans, upsample, sample')
p.add_argument('--DSR', default=5, type=int, help='downsample rate of model (i.e. 5 -> only use every 5 measurements')
args = p.parse_args()

# ===========
# Import Data
# ===========
dsr = args.DSR
param_dict = config_dict(args.config_file)
data = np.load(f'{args.data_file}',allow_pickle=True)[()]
# Training Data (downsampled)
V_train = data['Train']['V'][::dsr]
I_inj_train = data['Train']['I_inj'][::dsr]
# Validation Data (downsampled)
V_val = data['Validation']['V'][::dsr]
I_inj_val = data['Validation']['I_inj'][::dsr]

# =================================
# Get Time of Train/Validation Data
# =================================
t0,tn,dt = param_dict['Time'].values()
t = np.arange(t0,tn,dt*dsr)

# ======================
# Initialize RBF Network
# ======================
print('Initializing RBF Network...')
Y,X,TDE,centers = regression_matrices(V_train,I_inj_train,args.D_e,args.tau,args.n_centers,args.scale,center_method=args.clustering_method)

# =============
# Fit DDF Model
# =============
print('Fitting Model...')
model = RidgeCV(alphas = [.00001,.0001,.001,.01,.1,1,10,100,1000,10000], fit_intercept = False,cv = 10).fit(X,Y)
print(f'----> Number of model coefficients: {len(model.coef_)}')
print(f'----> Regularization parameter: {model.alpha_}')

# ==============
# Evaluate Model
# ==============
print('Predicting Future with Training Input...')
V_hat_train = predict_future(V_train,I_inj_train,args.tau,args.D_e,centers,model.coef_,args.scale)
# Get spike distances
spk_t_actual = rasterize(t,V_train)
spk_t_predict = rasterize(t,V_hat_train)
ref_spike_train = spk.SpikeTrain(spk_t_actual, [t0, tn])
ddf_spike_train = spk.SpikeTrain(spk_t_predict, [t0, tn])
isi_distance= spk.isi_distance(ref_spike_train, ddf_spike_train, interval=(t0, tn))
spike_distance= spk.spike_distance(ref_spike_train, ddf_spike_train, interval=(t0, tn))
print(f'----> ISI Distance:{isi_distance:.2f}')
print(f'----> Spike Distance:{spike_distance:.2f}')
plt.plot(t,V_train,alpha=0.7,color='darkcyan')
plt.plot(t,V_hat_train,alpha=0.7,color='black')
plt.vlines(spk_t_actual,60,65,color='darkcyan')
plt.vlines(spk_t_predict,70,75,color='black')
plt.show()

print('Predicting Future with Validation Input...')
V_hat_val = predict_future(V_val,I_inj_val,args.tau,args.D_e,centers,model.coef_,args.scale)
# Get spike distances
spk_t_actual = rasterize(t,V_val,threshold=10)
spk_t_predict = rasterize(t,V_hat_val,threshold=10)
ref_spike_train = spk.SpikeTrain(spk_t_actual, [t0, tn])
ddf_spike_train = spk.SpikeTrain(spk_t_predict, [t0, tn])
isi_distance= spk.isi_distance(ref_spike_train, ddf_spike_train, interval=(t0, tn))
spike_distance= spk.spike_distance(ref_spike_train, ddf_spike_train, interval=(t0, tn))
print(f'----> ISI Distance:{isi_distance:.2f}')
print(f'----> Spike Distance:{spike_distance:.2f}')
plt.plot(t,V_val,alpha=0.7,color='darkcyan')
plt.plot(t,V_hat_val,alpha=0.7,color='black')
plt.vlines(spk_t_actual,60,65,color='darkcyan')
plt.vlines(spk_t_predict,70,75,color='black')
plt.show()

# =========
# Save data
# =========
print(f'Saving data to: {args.out_path}')
data_dict = {
        'Model parameters': {
        'coeffs': model.coef_,
        'centers': centers,
        'D_e':args.D_e,
        'tau':args.tau,
        'scale':args.scale
        },
        'Time':{
        'downsample factor': args.DSR,
        'delta_t': dt*args.DSR,
        't0': t0,
        'tn': tn
        }
}
np.save(args.out_path,data_dict)

