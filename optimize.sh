#!/bin/sh 
### General options 
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J train_mtrl
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16 
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u vvipu@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 
# # OpenMP settings for parallel data collection
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# # Clear old GCC library paths that might conflict
# unset LD_LIBRARY_PATH
# # Or alternatively, ensure the new GCC libs are prioritized:
# # export LD_LIBRARY_PATH=/appl9/gcc/13.3.0-binutils-2.42/lib64:$LD_LIBRARY_PATH

# # Activate conda environment
cd ~
source ~/miniconda3/bin/activate
source ~/.bashrc
conda activate uav

# # Set CMAKE_PREFIX_PATH for the conda environment
# export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH

# # Change to simulator directory
cd ~/uav_reinforcement_learning_control

# Run training
python3 optimize.py --n-trials 100 --n-timesteps 1_000_000 --n-envs 16 --study-name ppo_wayp_track --storage sqlite:///optuna_wayp_track.db

echo "Optimization completed"
