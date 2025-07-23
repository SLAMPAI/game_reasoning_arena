#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --account=cstdl
#SBATCH --time=4:00:00
#SBATCH --job-name=debug_session

# This is an example of a persistent job that we can use to debug our code in an interactive way.

# The idea is to start this in tmux
# submit the job: sbatch persistent_job.sh
# attach to it: srun --jobid=<your_job_id> --pty bash
# then you can run your commands in the job

#module load <necessary_modules>
echo "Use: srun --jobid=$SLURM_JOB_ID --pty bash"
sleep infinity
