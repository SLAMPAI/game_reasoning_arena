
export DEBUG=1  # Enable debugging mode

# Get new keys

export PYTHONPATH="/p/project/ccstdl/cipolina-kun1/open_spiel_arena:$PYTHONPATH"

# Set game names as an environment variable
export GAME_NAMES="kuhn_poker,matrix_rps,tic_tac_toe,connect_four"
# Relevant paths
export PYTHONPATH="/p/project/ccstdl/cipolina-kun1/open_spiel_arena:$PYTHONPATH"
export MODELS_DIR="/p/data1/mmlaion/marianna/models"
export MODEL_CONFIG_FILE="/p/project/ccstdl/cipolina-kun1/open_spiel_arena/configs/models.json"


export VLLM_ATTENTION_BACKEND="FLASHINFER"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export GLOO_SOCKET_IFNAME=ib0
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export NCCL_NET_GDR_LEVEL=0

export OPENBLAS_NUM_THREADS=1  # en caso de que haya muchos threads open
export OMP_NUM_THREADS=1

export VLLM_PLATFORM="cpu"  # Importante para VLLM!

#module load CUDA
#module load Stages/2025 Python/3.12.3  #nodes have different python versions
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# For longer evaluations
export PYDEVD_WARN_EVALUATION_TIMEOUT=10



#Para que vaya mas rapido
ray stop  # Stop any previous Ray instances
ray start --head --num-cpus=8 --num-gpus=1 # Start a fresh local Ray cluster

# Without this it gives weird no file errors
chmod u+r /p/project/ccstdl/cipolina-kun1/open_spiel_arena/scripts/simulate.py

source /p/scratch/laionize/cache-kun1/miniconda3/bin/activate /p/scratch/laionize/cache-kun1/llm

python -m debugpy --listen 0.0.0.0:9000  /p/project/ccstdl/cipolina-kun1/open_spiel_arena/scripts/simulate.py
