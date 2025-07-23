#!/bin/bash

# To run:
# cd open_spiel_arena/slurm_jobs
# sbatch --export=EXPERIMENT_MODE="all" slurm_jobs/run_simulation.sh
# export DEBUG=1  # Enable debugging mode
# sbatch slurm_jobs/run_simulation.sh.sh


# ========================================
# SLURM Job Configuration
# ========================================
#SBATCH --nodes=1  #4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --job-name=llm_arena
#SBATCH --partition=booster
#SBATCH --account=laionize
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --output=/p/project/ccstdl/cipolina-kun1/open_spiel_arena/results/%x_%j.out

module load cuda

# ========================================
# Set Experiment Parameters
# ========================================
TEMPERATURE=0.3
MAX_MODEL_LENGTH=4096
N_TRIALS=1  #10  #TODO: delete all the commented parts below!
MAX_TOKENS=2038  # TODO: this is not used! see where mariana, was using it!

EXPERIMENT_MODE="llm_vs_random"  # Options: "llm_vs_random", "llm_vs_llm", "all"

# Environment Variables for vLLM and Ray
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export GLOO_SOCKET_IFNAME=ib0
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export NCCL_NET_GDR_LEVEL=0

# For longer evaluations
export PYDEVD_WARN_EVALUATION_TIMEOUT=10

# Things that have worked for me
#module load CUDA
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#module load Stages/2025 Python/3.12.3
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

module load Stages/2025 Python/3.12.3  #nodes have different python versions

# Debug Mode (set to 1 for debugging)
export DEBUG=1  # Change to 1 for debugging

# ========================================
# Retrieve the list of allocated nodes
# ========================================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes[0]}"  # First node as head
#nodes_array=($nodes)
#head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#head_node_ip="$(nslookup "${head_node}i" | grep -oP '(?<=Address: ).*')"

# Get the head node's IP
head_node_ip=$(ssh "$head_node" hostname -I | awk '{print $1}')
if [[ -z "$head_node_ip" ]]; then
    echo "Error: Failed to get head node IP!"
    exit 1
fi
echo "Head node IP: $head_node_ip"


# For one node start ray like this
#ray start --head --port=6379  --num-cpus=32  --temp-dir=/p/home/jusers/cipolina-kun1/juwels/ray_tmp --include-dashboard=False

# Define the port for ray communication
port= 6379 #20156
export RAY_ADDRESS="$head_node_ip:$port"

# ========================================
# Start the Ray Head Node
# ========================================
ray stop   # Ensure no previous Ray instances are running
echo "Starting HEAD at $head_node ($head_node_ip)"
srun --nodes=1 --ntasks=1 -w "$head_node" --gres=gpu:4 \
    ray start --head --port=$port \
    --num-gpus=$SLURM_GPUS_PER_NODE --num-cpus=${SLURM_CPUS_PER_TASK} --block &

#srun --nodes=1 --ntasks=1 -w "$head_node" --gres=gpu:4  \
#    ray start --head --node-ip-address="$head_node_ip" --port=$port \
#    --num-gpus $SLURM_GPUS_PER_NODE --num-cpus ${SLURM_CPUS_PER_TASK} --block &

sleep 10  # Allow time for initialization

# ========================================
# Start Ray Worker Nodes
# ========================================

# Reduce threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ulimit -u 8192  # Increase max user processes


worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node=${nodes[$i]}

    # Get worker node's IP (safer method)
    node_ip=$(ssh "$node" hostname -I | awk '{print $1}')
    if [[ -z "$node_ip" ]]; then
        echo "Error: Failed to get IP for worker $i ($node)"
        continue
    fi

    echo "Starting WORKER $i at $node ($node_ip)"
    srun --nodes=1 --ntasks=1 -w "$node" --gres=gpu:4 \
        ray start --address "$RAY_ADDRESS" \
        --num-cpus=${SLURM_CPUS_PER_TASK} --num-gpus=$SLURM_GPUS_PER_NODE --block &

    sleep 5  # Allow each worker to initialize
done

# Check Ray status
ssh "$head_node" ray status

#for ((i = 1; i <= worker_num; i++)); do
#    node=${nodes_array[$i]}
#    node_ip="$(nslookup "${node}i" | grep -oP '(?<=Address: ).*')"

#    echo "Starting WORKER $i at $node"
#    srun --nodes=1 --ntasks=1 -w "$node" --gres=gpu:4 \
#        ray start --address "$RAY_ADDRESS" \
#        --node-ip-address="$node_ip"  \
#        --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus $SLURM_GPUS_PER_NODE --block &
#    sleep 10  # Allow time for initialization
#done

# ========================================
# Directories
# ========================================

# Set game names as an environment variable
export GAME_NAMES="kuhn_poker,matrix_rps,tic_tac_toe,connect_four"

# Relevant paths
export PYTHONPATH="/p/project/ccstdl/cipolina-kun1/open_spiel_arena:$PYTHONPATH"
export MODELS_DIR="/p/data1/mmlaion/marianna/models"
export MODEL_CONFIG_FILE="/p/project/ccstdl/cipolina-kun1/open_spiel_arena/configs/models.json"

# Activate Micromamba environment
export MAMBA_ROOT_PREFIX=/p/scratch/laionize/cache-kun1/mamba
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s bash)"
micromamba activate llm

# Project directory
cd /p/project/ccstdl/cipolina-kun1/open_spiel_arena

# ========================================
# Run the Simulation with Auto Mode Selection
# ========================================
if [[ "$EXPERIMENT_MODE" == "all" ]]; then
    echo "Running all experiments: LLM vs Random and LLM vs LLM"
    python3 scripts/simulate.py --mode llm_vs_random --config configs/config.json
    python3 scripts/simulate.py --mode llm_vs_llm --config configs/config.json
elif [[ "$EXPERIMENT_MODE" == "llm_vs_random" ]]; then
    echo "Running: LLM vs Random"
    python3 scripts/simulate.py --mode llm_vs_random --config configs/config.json
elif [[ "$EXPERIMENT_MODE" == "llm_vs_llm" ]]; then
    echo "Running: LLM vs LLM"
    python3 scripts/simulate.py --mode llm_vs_llm --config configs/config.json
else
    echo "ERROR: Invalid mode specified. Use 'llm_vs_random', 'llm_vs_llm', or 'all'."
    exit 1
fi
