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

MODEL_NAME=${1:-CLIP}
# You might get ann OOM error if you try all at the same time (going from RAVEN-10000 directly)
# Intstead just pick the dir you want and fire off a couple batches at the same time 
# if you are really impatient (each should take ~3 mins depending on which gpu get)
DATA_DIR=${2:-"/oscar/scratch/$USER/RAVEN-10000/center_single"}
OUTPUT_DIR=${3:-"/oscar/scratch/$USER/embeddings/clip-single"}

echo "Model: ${MODEL_NAME}"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Force unbuffered output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH=~/scratch/spatial-probing

module purge
unset LD_LIBRARY_PATH
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda deactivate
conda activate team2d
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
PYTHON_EXEC="/oscar/home/$USER/.conda/envs/team2d/bin/python"
# Use the correct pre-built container path (note: x86_64.d not x86_64)
#CONTAINER_PATH="/oscar/home/$USER/scratch/pytorch-24.11-py3.simg"
#EXEC_PATH="srun apptainer exec --nv"
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

#echo "GPU Information (inside container):"
#$EXEC_PATH $CONTAINER_PATH nvidia-smi
#echo ""

echo "PyTorch GPU Detection:"
python -c "import torch; print('PT version:', torch.__version__); print('Built with CUDA:', torch.backends.cuda.is_built()); print('GPUs detected:', torch.cuda.device_count()); print('GPU devices:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
echo ""

echo "=========================================="
echo "Installing dependencies"
echo "=========================================="
echo ""

#$EXEC_PATH $CONTAINER_PATH pip install --user --no-cache-dir tqdm

echo ""
echo "=========================================="
echo "Starting main Python script at $(date)"
echo "=========================================="
echo ""

cd "${SLURM_SUBMIT_DIR}" || exit 1
echo "Working directory: $(pwd)"
echo ""

python -u extract_embeddings.py \
--model $MODEL_NAME \
--data_dir $DATA_DIR \
--save_dir $OUTPUT_DIR

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Python script finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE

