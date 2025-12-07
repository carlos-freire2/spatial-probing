#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -p gpu --gres=gpu:1     # how many gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 02:30:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=32000MB           # INCREASED from 16GB to 32GB
#SBATCH --job-name='Embeddings'
#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err

# Force unbuffered output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

module purge
unset LD_LIBRARY_PATH
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
# Use the correct pre-built container path (note: x86_64.d not x86_64)
CONTAINER_PATH="/oscar/home/$USER/scratch/pytoch-25.11-py3.simg"
EXEC_PATH="srun apptainer exec --nv"
echo ""
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

echo "GPU Information (from host):"
nvidia-smi
echo ""

echo "GPU Information (inside container):"
$EXEC_PATH $CONTAINER_PATH nvidia-smi
echo ""

echo "PyTorch GPU Detection:"
$EXEC_PATH $CONTAINER_PATH python -c "import torch; print('PT version:', torch.__version__); print('Built with CUDA:', torch.backends.cuda.is_built()); print('GPUs detected:', torch.cuda.device_count()); print('GPU devices:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
echo ""

echo "=========================================="
echo "Installing dependencies"
echo "=========================================="
echo ""

$EXEC_PATH $CONTAINER_PATH pip install --user --no-cache-dir tqdm

echo ""
echo "=========================================="
echo "Starting main Python script at $(date)"
echo "=========================================="
echo ""

cd "${SLURM_SUBMIT_DIR}" || exit 1
echo "Working directory: $(pwd)"
echo ""

embedding_dir="embeddings/"
save_dir="experiments/"
models=("DINOv3" "CLIP")
layers=(2 4 6 8 10 12)
probes=(linear MLP)
dataset_name="RAVEN"
data_configs="center_single"
tasks=("Type" "Size" "Color")
seed=42
lr=0.001
batch_size=64
epochs=15

for model in "${models[@]}"; do
    for layer in "${layers[@]}"; do
        for probe in "${probes[@]}"; do
            for task in "${tasks[@]}"; do
                python scripts/run_experiment.py --embedding_dir "$embedding_dir" \
                                                --save_dir "$save_dir" \
                                                --model "$model" \
                                                --layer "$layer" \
                                                --probe "$probe" \
                                                --dataset_name "$dataset_name" \
                                                --data_config "$config" \
                                                --task "$task" \
                                                --seed "$seed" \
                                                --lr "$lr" \
                                                --batch_size "$batch_size" \
                                                --epochs "$epochs"
            done
        done
    done
done

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Python script finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
