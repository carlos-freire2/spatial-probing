"""
Runs a single probe experiment for the specified conditions.
Saves probe.pt, the trained probe weights, and metrics.json, the final metrics 
and model configs, in the directory defined by save_dir/model/layer/probe/dataset_name/task/. 

Example: 
python scripts/run_experiment.py \
    --embedding_dir embeddings/RAVEN/DINOv3/ \
    --save_dir experiments/ \
    --model DINOv3 \
    --layer 1 \
    --probe linear \
    --dataset_name RAVEN \
    --task shape \
    --seed 42 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 10 
"""

def main():
    # TODO define argparse arguments
    # TODO load embeddings, organize for task
    # TODO initialize and train probe (use probing/mlp_probe.py or linear_probe.py, train.py)
    # TODO save model weights and metrics (append model config parameters to metrics)
    raise NotImplementedError()

if __name__ == "__main__":
    main() 