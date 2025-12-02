"""
Generates hidden-state embeddings for the specified model and data.
Saves embeddings as .npy files of shape (N, D), one per hidden layer.

Example: 
python scripts/generate_embeddings.py \
    --model DINOv3 \
    --data_dir data/RAVEN \
    --save_dir embeddings/RAVEN/DINOv3 \
    --batch_size 16
"""

def main():
    # TODO define argparse arguments
    # TODO pass data to model (use models/clip_model.py or dino_model.py)
    # TODO save embeddings for each layer
    raise NotImplementedError()

if __name__ == "__main__":
    main() 
