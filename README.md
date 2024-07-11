# Overview
This repo contains the relevant source code to reproduce the analyses in Fehrman and Meliza (2024).

We controlled the voltage of a Connor-Stevens (CS) conductance-based neuron model using nonlinear model predictive control (MPC). While the parameters of the CS model in priniciple can be emperically estimated, 
we simulated the case where very little knowledge of the controlled neuron is known, and the dynamics must be approximated using data-driven methods. This was inspired by the data-driven forecasting (DDF) model proposed
by Clark et al (2022). We used two dynamically distinct CS models (Type-I, Type-II) and demonstrated that they can be controlled with MPC without needing to know much of the biophysical details of the neurons.

This README will outline how reproduce our results. Almost all scripts used command line arguments to streamline iterating alternative parameter values.

**IMPORTANT** Some users may experience an error when trying to install `pyspike`. One solution is to clone the source code and do a local setup:
- `git clone https://github.com/mariomulansky/PySpike.git`
- `cd PySpike`
- `python setup.py build_ext --inplace`

Then add the path to `PySpike` to the Python environment variable (e.g. `sys.path.append(**path to PySpike**)`). These instructions are taken from the authors of `pyspike`
and can be found here `https://mariomulansky.github.io/PySpike/`.

# Simualting a Connor-Stevens Neuron
The scripts necessary to simulate the activity of a CS neuron when stimulated with a known injected current ($I_{inj}$) and unknown noise current ($I_{noise}$) can be found in `neuron_scripts` with the configuration parameters found in `config_files`.
## `neuron_scripts`
### `connor_stevens.py`
Contains the differential equations of the CS model along with a check to see if it is a Type-I or Type-II model based on the config file used.

### `poisson_spiking.py`
Contains scripts for basic random Poisson spiking used for $I_{noise}$. 

### `neuron_inputs.py`
Used to create both the known $I_{inj}$ and $I_{noise}$ currents. Additionally, has functions for scaling the signal-to-noise ratio (SNR) of $I_{inj}$ and $I_{noise}$.

### `stimulate_neuron.py`
Given a pre-computed $I_{inj}$ and $I_{noise}$, simulates the membrane voltage of the CS neuron models.

### `waveform_analysis.py`
Used to convert an array of membrane voltage in a spike train. Also contains functionality for extracting the mean spiking waveform used in experiment 3 of the manuscript.

# Constructing a Data-Driven Forecasting Model of the Connor-Stevens Neuron
In our manuscript, we simulate a series of experiments where a known injected current is applied to a CS neuron. The resulting membrane voltages are used along with the injected current to build a discrete-time DDF model. The scripts necessary to do this are found in
`DDF_scripts`.

## `DDF_scripts`

### `DDF.py`
Following Clark et al (2022), we used a radial basis function network (RBFN) for the DDF model. This model uses time-delay embedding of membrane voltage to predict the voltage at the next time step. All required scripts to construct this DDF model can be found here.

### `generate_assimilation_data.py`
The $I_{inj}$ and $I_{noise}$ currents used for generating the DDF training and validation data are found here. **If you are attempting to reproduce our results, this is the first script you should run.**

### `optimal_embedding.py`
Using the Simplex method proposed by Sugihara (1990), the `pyEDM` package is used to find the optimal time-delay embedding dimension for the DDF model.

### `train_DDF_model.py`
Stimulates a given CS neuron with the training $I_{inj}$ and $I_{noise}$ currents produced in `generate_assimilation_data.py`. The $I_{inj}$ and resulting voltages are downsampled (DSR in the config file)
and used to train the DDF model via ridge regression. The validation currents are then used to indicate model fit. **If you are attempting to reproduce our results, this is the second script you should run.**

# MPC of Connor-Stevens Neuron
Using the DDF model, we can now use the `do-mpc` package to perform nonlinear MPC of the CS neuron. The scripts for generating reference trajectories and implementing MPC are found in `mpc_scripts`.
It is strongly encouraged to read the official documentation of `do-mpc` in order to understand much of the code here.

## `mpc_scripts`

### `generate_reference_trajectories.py`
Repeatedly stimulates a given CS neuron with injected and noise currents and saves the resulting membrane voltages as reference trajectories to use in MPC. 
**If you are attempting to reproduce our results, this is the third script you should run.**

### `casadi_rbf.py`
Useful function that converts the Gaussian nonlinearity of the RBFN into a CasADI object that can be used by the optimizer in `do-mpc`.

### `mpc_of_neuron.py`
For a given CS neuron, DDF model, experiment number, and reference trajectory, controls the voltage of the CS neuron with MPC. The hyperparameters of the controller are given as command line arguments. 
**If you are attempting to reproduce our results, this is the fourth script you should run.**

### `analyze_mpc_performance.py`
Loops through saved MPC data and evaluates how well the controller performed. **If you are attempting to reproduce our results, this is the fifth script you should run.**

# Open-Loop Control of Connor-Stevens Neuron
In our manuscript, we compared the performance of MPC to the open-loop method of repeating $I_{inj}$ while varying $I_{noise}$ in experiments 1 and 2. In experiment 3, we used a pulse current for open-loop stimulation. 
It also contains code to further evaluate the DDF model by comparing its predictions to repeated stimulations of the CS neuron with varying $I_{noise}$.
The scripts for this can be found in `open_loop_scripts`.

## `open_loop_scripts`

### `create_null_distribution.py`
Using the validation $I_{inj}$ obtained from `generate_assimilation_data.py`, a given CS neuron is repeated stimulated with the injected current but varying $I_{noise}$ currents. 

### `perfect_model.py`
This is used in tandem with the results of `create_null_distribution.py`. It stimulates a given CS neuron only with the validation $I_{inj}$, but without $I_{noise}$. This gives an upper bound on how good the DDF model could possibly be.

### `open_loop_control.py`
For a given CS neuorn, experiment number, and reference trajectory, controls the voltage of the CS neuron with open-loop control. For experiment 3, a pulse width and amplitude parameter is required.
**If you are attempting to reproduce our results, this is the sixth script you should run.**

### `analyze_open_loop_performance.py`
Loops through saved open-loop data and evaluates how well the controller performed. **If you are attempting to reproduce our results, this is the seventh script you should run.**

# Proportional Control of Connor-Stevens Neuron
In experiment 3, the control performance of MPC, open-loop control with a pulse, and proportional feedback control is compared. The script to do this can be found in `p_controller_scripts/p_control.py`. Only a single trial was used in the manuscript.
The controller gain and reference trajectory are specified with command line arguments. Additionally, upper and lower bounds to the control signal can be added as a fair comparison to the contraints used in `mpc_of_neuron.py`. 
**If you are attempting to reproduce our results, this is the eigth script you should run.**

# Constructing Figures
Many of the figures in the manuscript can be reproduced by running the above scripts and then followed by running the scripts in `figure_scripts`.

