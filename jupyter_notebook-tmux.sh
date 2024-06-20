#!/bin/bash
#
# CompecTA (c) 2019
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Slurm" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Set the required time limit for the job with --time parameter.
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch slurm_example.sh

# -= Resources =-
#
# #SBATCH --ntasks=16
# #SBATCH --mem=64G
#SBATCH --job-name=jupytert
#SBATCH --account=mdbf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --nodelist=cn08
#SBATCH --time=14-00:00:00
#SBATCH --output=unsloth-%j.out
#SBATCH --gres=gpu:1
# #SBATCH --gres=gpu:tesla_v100:1

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

echo ""
echo "======================================================================================"
# env
echo "======================================================================================"
echo ""

# Set stack size to unlimited
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

# Activate Python virtual environment
eval "$(conda shell.bash hook)"
conda activate unsloth_env

echo "Running Jupyter Notebook..!"
echo "==============================================================================="

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

### print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "

    MacOS or linux terminal command to create your ssh tunnel
    -----------------------------------------------------------------
    ssh -N -L ${port}:${node}:${port} ${user}@${cluster}10.3.5.102
    -----------------------------------------------------------------

    Windows MobaXterm info
    Forwarded port:same as remote port
    Remote server: ${node}
    Remote port: ${port}
    SSH server: ${cluster}10.3.5.102
    SSH login: $user

    Use a Browser on your local machine to go to:
    localhost:${port}  (prefix w/ https:// if using password)
"


# Start Jupyter Notebook in a tmux session and redirect output directly to the Slurm output file
tmux new -d -s jupyter_session "jupyter-notebook --no-browser --port=${port} --ip=${node} >> unsloth-${SLURM_JOB_ID}.out 2>&1"

echo "Jupyter Notebook is running inside a tmux session named 'jupyter_session', tmux is now detached."
echo "The job on the machine will remain active as long as the tmux session is alive until the allocated time is over."
# Loop to check if the tmux session is still alive
while tmux has-session -t jupyter_session 2>/dev/null; do
    sleep 10  # Wait for 10 seconds before checking again
done
