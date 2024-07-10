import sys
sys.path.append('neuron_scripts')
sys.path.append('DDF_scripts')
from DDF import *
from waveform_analysis import rasterize
from connor_stevens import *
import argparse
import matplotlib.pyplot as plt
from casadi_rbf import PSI
import do_mpc
import casadi as ca
from neuron_inputs import *
import numpy as np
from scipy.integrate import odeint

'''
Run MPC on CS neuron using DDF model
'''

# =======================
# Parse Command Line Args
# =======================
p = argparse.ArgumentParser()
p.add_argument('--config_file', default='config/config_cs_type_II.yaml', type=str, help='file path to config file for assimilation data')
p.add_argument('--ddf_path', default = 'saved_models/ddf_type_II.npy', type = str, help = 'path to DDF model.')
p.add_argument('--out_path',default = 'control_output', type = str, help = 'path to where output is to be saved.')
p.add_argument('--exp_num',default=1,type=int,help='which of the three experiments of manuscript to reproduce')
p.add_argument('--trial_indx',default =0, type = int, help = 'which trial to use in reference data.')
p.add_argument('--n_horizon', default = 5, type = int, help = 'how many steps into the future the MPC algorithm should optimize over')
p.add_argument('--Q', default = 5, type = float, help = 'scaling for errors state cost')
p.add_argument('--S', default = 1, type = float, help = 'scaling for final state error cost')
p.add_argument('--R', default = 50, type = float, help = 'scaling for penalty in input fluctuation')
p.add_argument('--u_UB', default = 100.0, type = float, help = 'upper bound of input (constraint)')
p.add_argument('--u_LB', default = -100.0, type = float, help = 'lower bound of input (constraint)')
args = p.parse_args()

# ================
# Construct Neuron
# ================
neuron = construct_neuron(args.config_file)
param_dict = config_dict(args.config_file)
sim_type = param_dict['Name']['sim type']

# ==============
# Load DDF Model
# ==============
ddf_data = np.load(args.ddf_path,allow_pickle=True)[()]
coeffs,centers,D_e,tau,scale = ddf_data['Model parameters'].values()
# Separate coefficients
W = coeffs[:-1]
alpha = coeffs[-1]

# =============================
# Get Real Time and Sample Time
# =============================
t0,tn,dt = param_dict['Time'].values()
dsr,delta_t,_,_ = ddf_data['Time'].values()
t = np.arange(t0,tn,dt)
T = np.arange(t0,tn,delta_t)

# ===========================================
# Load Reference Trajectory and Noise Current
# ===========================================
if args.exp_num == 1:
    ref_data = np.load(f'reference_trajectories/type_{sim_type}_ref_trajectories.npy',allow_pickle=True)[()]
    old_noise = ref_data['I_noise'][args.trial_indx]
elif args.exp_num == 2:
    if sim_type == 'I':
        ref_type = 'II'
    else:
        ref_type = 'I'
    ref_data = np.load(f'reference_trajectories/type_{ref_type}_ref_trajectories.npy',allow_pickle=True)[()]
    old_noise = np.load(f'reference_trajectories/type_{sim_type}_ref_trajectories.npy', allow_pickle=True)[()]['I_noise'][args.trial_indx]
elif args.exp_num == 3:
    ref_data = np.load(f'reference_trajectories/type_{sim_type}_waveform_ref_trajectories.npy',allow_pickle=True)[()]
    old_noise = np.load(f'reference_trajectories/type_{sim_type}_ref_trajectories.npy', allow_pickle=True)[()]['I_noise'][args.trial_indx]

# Get noise current parameters
_, tau_noise, g_syn_max, spiker_fr = param_dict['Noise Current'].values()
# Generate new noise input
I_noise = balanced_noise_current(spiker_fr, dt, t, g_syn_max, tau_noise)
# Scale SNR of I_noise to be same as noise that produced sim_type reference trajectory (i.e. equivalent power)
I_noise = scale_noise_to_SNR(old_noise[:len(I_noise)],I_noise,1)

# Downsample for DDF and add holding values at end for horizon optimization
ref_traj = ref_data['V'][args.trial_indx][:len(t)][::dsr]
ref_traj = np.concatenate((ref_traj,np.repeat(ref_traj[-1],args.n_horizon)))

# ========================
# Initialize Control Input
# ========================
i_now = np.array([0]).reshape(1,1)

# =================================
# Set Up DDF Model for Optimization
# =================================
DDF_model = do_mpc.model.Model('discrete')
# State variables (V_n: V now, V_d1: V with one time delay (ie previous value))
V_n = DDF_model.set_variable(var_type = '_x', var_name = 'V_n', shape = (1,1))
Vd1 = DDF_model.set_variable(var_type = '_x', var_name = 'Vd1', shape = (1,1))
# Input to optimize
I_future = DDF_model.set_variable(var_type = '_u', var_name = 'I_future', shape = (1,1))
# Time-varying parameters
I_now = DDF_model.set_variable(var_type = '_tvp', var_name = 'I_now', shape = (1,1))
V_ref = DDF_model.set_variable(var_type = '_tvp', var_name = 'V_ref', shape = (1,1))
# Define DDF Equations
# Model: V[n+1] = V[n]+dot(W,Psi[n])+alpha*(I[n+1]+I[n])
psi = ca.vertcat(*PSI(ca.vertcat(V_n,Vd1),centers,scale)) # this converts the tuple to a CASADI SX object
DDF_model.set_rhs('V_n',V_n+ca.dot(W,psi)+alpha*(I_future+I_now))
DDF_model.set_rhs('Vd1',V_n)
DDF_model.setup()

# =================
# Set up controller
# =================
mpc = do_mpc.controller.MPC(DDF_model)
suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
controller_params = {'n_horizon': args.n_horizon, 't_step': delta_t, 'nlpsol_opts' : suppress_ipopt,'n_robust': 0,'store_full_solution':True}
mpc.set_param(**controller_params)
final_state_cost = (DDF_model.x['V_n']-DDF_model.tvp['V_ref'])**2
running_cost = (DDF_model.x['V_n']-DDF_model.tvp['V_ref'])**2
mpc.set_objective(lterm = args.Q*running_cost,mterm = args.S*final_state_cost)
mpc.set_rterm(I_future = args.R)
mpc.bounds['lower','_u', 'I_future'] = args.u_LB
mpc.bounds['upper','_u', 'I_future'] = args.u_UB

# ============================
# Time-Varying Parameters Loop
# ============================
tvp_struct_mpc = mpc.get_tvp_template()
# The below bool is due to do-mpc calling the tvp_func_mpc when compiling and does not like
# that I_future is not set yet. Essentially this is a problem due to the non-standard control
# problem of having both u(n) and u(n+1) in the state-space model.
running_optimization = False
def tvp_fun_mpc(t_now):
    time_indx = int(np.round(t_now/delta_t))
    tvp_struct_mpc['_tvp',0,'V_ref'] = ref_traj[time_indx]
    tvp_struct_mpc['_tvp',0,'I_now'] = i_now
    if running_optimization == True:
        for k in range(1,args.n_horizon+1):
            tvp_struct_mpc['_tvp',k,'V_ref'] = ref_traj[k+time_indx]
            tvp_struct_mpc['_tvp',k,'I_now'] = mpc.data.prediction(('_u','I_future'))[0][k-1]
    return tvp_struct_mpc
mpc.set_tvp_fun(tvp_fun_mpc)
mpc.setup()

# ===================================================
# Burn in of CS model to get values to seed DDF model
# ===================================================
# burn_in: how many integration iterations are needed to seed the model
burn_in = int((D_e-1)*tau*delta_t/dt)
# V_actual: collector for CS neuron voltage
V_actual = np.zeros_like(I_noise)
# Get initial conditions
X0 = np.random.uniform(0, 1, 6)
V_actual[0] = -72.8 + np.random.normal(0, 1, 1)
X0[0] = V_actual[0]
# Stimulate with noise only
for i in range(1,burn_in+1):
    tspan = [t[i - 1], t[i]]
    X = odeint(neuron.ode_eqs, X0, tspan, args=(I_noise[i],))
    X0 = X[1]
    V_actual[i] = X0[0]
# V_control: the discrete time collector of voltage
V_control = np.zeros_like(ref_traj)
V_control[:D_e] = V_actual[:burn_in+1][::dsr]

# Collect DDF predictions
V_DDF = np.zeros_like(ref_traj)
V_DDF[:D_e] = V_control[:D_e]

# =========================================
# Initialize controller with burn in values
# =========================================
mpc.x0['V_n'] = V_control[D_e-1]
mpc.x0['Vd1'] = V_control[0]
x0 = mpc.x0
mpc.set_initial_guess()

# =============================================================================
# Set up initial control value and empty collector for optimized control inputs
# =============================================================================
I_control = np.zeros_like(t)
i_future = np.array([0]).reshape(1,1)

# ============
# Control Loop
# ============
len_of_loop = len(I_noise)
discrete_indx = 0 # Used to update DDF model which is downsampled from the time resolution of the simulation dt.
running_optimization=True
for i in range(burn_in,len_of_loop):
    # Update optimized inputs
    i_now = i_future
    I_control[i] = i_now[0][0]
    if i%dsr == 0:
        print(f'Iteration: {i}/{len_of_loop}')
        # Uncomment for no monitoring of true state -- DDF only predictions (does not work well)
        #V_DDF[discrete_indx+D_e] = single_step_prediction(V_DDF[discrete_indx+D_e-1],V_DDF[discrete_indx+D_e-2],i_now[0][0],i_future[0][0],centers,W,alpha,scale)
        #i_future = mpc.make_step(np.array(V_DDF[discrete_indx:D_e + discrete_indx]))
        i_future = mpc.make_step(np.array(V_control[discrete_indx:D_e+discrete_indx])) # mpc optimized input
    tspan = [t[i - 1], t[i]]
    # Inject current
    u = i_now+i_future
    # Numerically integrate one dt into the future
    X = odeint(neuron.ode_eqs, X0, tspan, args=(u[0][0]+I_noise[i],))
    X0 = X[1]
    V_actual[i] = X0[0]
    if i%dsr == 0:
        V_control[discrete_indx+D_e] = X0[0]
        discrete_indx += 1

'''
# Uncomment to see results of control
ref_spikes = rasterize(T,ref_traj)
control_spikes = rasterize(T,V_control)
fig,ax = plt.subplots(2,1,sharex=True)
ax[0].vlines(ref_spikes,60,65,color='black')
ax[0].vlines(control_spikes,70,75,color='red')
ax[0].plot(T,ref_traj[:len(T)],color='black',alpha=0.7)
ax[0].plot(T,V_control[:len(T)],color='red',alpha=0.7)
ax[1].plot(t,I_control,color='black',alpha=0.7)
ax[1].plot(t,I_noise,color='darkmagenta',alpha=0.7)
plt.show()
'''

# =========
# Save Data
# =========
out_file = f'{args.out_path}/exp_{args.exp_num}/type_{sim_type}_trial_indx_{args.trial_indx}.npy'
print(f'Saving data: {out_file}')
data_dict = {
    'reference trajectory':ref_traj,
    'V_actual': V_actual,
    'V_control': V_control,
    'I_control': I_control,
    'I_noise': I_noise,
    'Time': {
        'dt': dt,
        't0': t0,
        'tn': tn,
        'delta_t': delta_t
    }
}
np.save(out_file, data_dict)