#!/bin/bash
for i in {0..49}
do
echo $i
python mpc_scripts/mpc_of_neuron.py --trial_indx=$i --Q=5 --S=1 --R=7 --exp_num=2
done