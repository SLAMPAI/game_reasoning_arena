# ========================================
# SLURM Job Configuration
# ========================================
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --job-name=llm_arena
#SBATCH --partition=
#SBATCH --account=
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --output=/results/%x_%j.out

module load cuda

# ========================================
# Retrieve the list of allocated nodes
# ========================================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes[0]}"  # First node as head

# Get the head node's IP
head_node_ip=$(ssh "$head_node" hostname -I | awk '{print $1}')
if [[ -z "$head_node_ip" ]]; then
    echo "Error: Failed to get head node IP!"
    exit 1
fi
echo "Head node IP: $head_node_ip"


# For one node start ray like this
#ray start --head --port=6379  --num-cpus=32  --temp-dir=/p/home/ray_tmp --include-dashboard=False

# Define the port for ray communication
port= 6379
export RAY_ADDRESS="$head_node_ip:$port"

# ========================================
# Start the Ray Head Node
# ========================================
ray stop   # Ensure no previous Ray instances are running
echo "Starting HEAD at $head_node ($head_node_ip)"
srun --nodes=1 --ntasks=1 -w "$head_node" --gres=gpu:4 \
    ray start --head --port=$port \
    --num-gpus=$SLURM_GPUS_PER_NODE --num-cpus=${SLURM_CPUS_PER_TASK} --block &


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
