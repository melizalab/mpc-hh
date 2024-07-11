import sys
sys.path.append('neuron_scripts')
from connor_stevens import *
from stimulate_neuron import *
import argparse
from neuron_inputs import *
import numpy as np
from scipy.integrate import odeint
from neuron_inputs import *

'''
Proportional control of neural state. While only used for experiment 3 in paper, could also be used
for other experiments as well.
'''


# =======================
# Parse Command Line Args
# =======================
p = argparse.ArgumentParser()
p.add_argument('--config_file', default='config/config_cs_type_I.yaml', type=str, help='file path to config file for assimilation data')
p.add_argument('--out_path',default = 'p_control_output', type = str, help = 'path to where output is to be saved.')
p.add_argument('--exp_num',default=1,type=int)
p.add_argument('--trial_indx',default =0, type = int, help = 'which trial to use in reference data.')
p.add_argument('--Kp', default = 0, type = float)
p.add_argument('--u_UB', default = 100.0, type = float, help = 'upper bound of input (constraint)')
p.add_argument('--u_LB', default = -100.0, type = float, help = 'lower bound of input (constraint)')
args = p.parse_args()

# ================
# Construct Neuron
# ================
neuron = construct_neuron(args.config_file)
param_dict = config_dict(args.config_file)
sim_type = param_dict['Name']['sim type']

# =============================
# Get Real Time and Sample Time
# =============================
t0,tn,dt = param_dict['Time'].values()
t = np.arange(t0,tn,dt)
dsr = 5
delta_t = 0.1
T = np.arange(t0,tn,delta_t)

# ===========================================
# Load Reference Trajectory and Noise Current
# ===========================================
if args.exp_num == 1:
    ref_data = np.load(f'reference_trajectories/type_{sim_type}_ref_trajectories.npy',allow_pickle=True)[()]
    I_inj = ref_data['I_inj'][args.trial_indx]
    old_noise = ref_data['I_noise'][args.trial_indx]
elif args.exp_num == 2:
    if sim_type == 'I':
        ref_type = 'II'
    else:
        ref_type = 'I'
    ref_data = np.load(f'reference_trajectories/type_{ref_type}_ref_trajectories.npy',allow_pickle=True)[()]
    I_inj = ref_data['I_inj'][args.trial_indx]
    old_noise = np.load(f'reference_trajectories/type_{sim_type}_ref_trajectories.npy', allow_pickle=True)[()]['I_noise'][args.trial_indx]
elif args.exp_num == 3:
    ref_data = np.load(f'reference_trajectories/type_{sim_type}_waveform_ref_trajectories.npy',allow_pickle=True)[()]
    old_noise = np.load(f'reference_trajectories/type_{sim_type}_ref_trajectories.npy', allow_pickle=True)[()]['I_noise'][args.trial_indx]
#Noise Current
_, tau_noise, g_syn_max, spiker_fr = param_dict['Noise Current'].values()
# Generate new noise input
I_noise = balanced_noise_current(spiker_fr, dt, t, g_syn_max, tau_noise)
# Scale SNR of I_noise to be same as noise that produced reference trajectory (i.e. equivalent power)
I_noise = scale_noise_to_SNR(old_noise,I_noise,1)
# Downsample for DDF
ref_traj = ref_data['V'][args.trial_indx][::dsr]


# ===================================================
# Burn in of CS model to get values to seed DDF model
# ===================================================
# burn_in: needed for DDF model in MPC, so this is here for a fair comparison.
burn_in = 5
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
V_control[:2] = V_actual[:burn_in+1][::dsr]

# =============================================================================
# Set up initial control value and empty collector for optimized control inputs
# =============================================================================
# Initialize control input
I_control = np.zeros_like(t)
def clamp(x, minimum=args.u_LB, maximum=args.u_UB):
    # Clamp values to be identical to MPC constraints
    return max(minimum, min(x, maximum))

# Length of simulation should be same as control input
len_of_loop = len(I_control)

# Get initial state error
state_error = ref_traj[0]-V_actual[0]

# Initialize collector for state errors.
errors = np.zeros_like(T)
errors[0] = state_error

# Get first P-controller input
u = args.Kp*state_error

# Next reference trajectory index
discrete_indx = 1

# Simulate!!!
for i in range(1,len_of_loop):
    if i%dsr == 0: # Only check state error every dsr time steps
        #print(f'Iteration: {i}/{len_of_loop}')
        state_error = ref_traj[discrete_indx-1]-V_actual[i-1]
        # New debug
        errors[discrete_indx] = state_error
        u = clamp(args.Kp*state_error)
        discrete_indx+=1
    I_control[i] = u # update I_control every time step
    tspan = [t[i-1], t[i]]
    X = odeint(neuron.ode_eqs, X0, tspan, args=(I_control[i]+I_noise[i],))
    X0 = X[1]
    V_actual[i] = X0[0]

'''
# Uncomment to see performance
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(T, ref_traj, color='black', alpha=0.7,label = 'ref traj')
ax[0].plot(t, V_actual, color='red', alpha=0.7, label = 'V actual')
ax[1].plot(t, I_control, color='blue', alpha=0.7,label='Control Input')
ax[1].plot(t, I_noise, color='darksalmon', alpha=0.7,label='Noise Input')
ax[0].legend()
ax[1].legend()
ax[0].set_title('State Variables')
ax[1].set_title('Control Input')
plt.show()
breakpoint()
'''
# =========
# Save Data
# =========
out_file = f'{args.out_path}/exp_{args.exp_num}/type_{sim_type}_trial_indx_{args.trial_indx}.npy'
print(f'Saving data: {out_file}')
data_dict = {
    'reference trajectory':ref_traj,
    'V_actual': V_actual,
    'I_control': I_control,
    'I_noise': I_noise,
    'Time': {
        'dt': dt,
        't0': t0,
        'tn': tn,
        'delta_t':delta_t,
        'dsr':dsr
    }
}
np.save(out_file, data_dict)