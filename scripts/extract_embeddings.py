"""
Extracts hidden-state embeddings for the specified model and data.
Expects data directory to be a folder of images.
Saves embeddings as .npy files of shape (N, D), one per hidden layer.

Example: 
python scripts/extract_embeddings.py \
    --model DINOv3 \
    --data_dir /data/RAVEN \
    --save_dir /embeddings/RAVEN/DINOv3 
"""
import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import argparse
from models.dino_model import DINOv3Wrapper
from models.clip_model import CLIPWrapper
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate hidden state embeddings")
    parser.add_argument("--model", choices=["DINOv3", "CLIP"], help="model, options are DINOv3 or CLIP")
    parser.add_argument("--data_dir", help="path to directory where image files are located")
    parser.add_argument("--save_dir", help="path to directory where embeddings will be saved")
    args = parser.parse_args()

    # load images 
    images = []
    for filename in os.listdir(args.data_dir):
        if filename.startswith("."): continue
        images.append(Image.open(os.path.join(args.data_dir, filename)).convert('RGB'))

    # get hidden state outputs from the selected model
    if args.model == "DINOv3":
        model = DINOv3Wrapper(model_name="facebook/dinov3-vits16-pretrain-lvd1689m")
    if args.model == "CLIP":
        model = CLIPWrapper(model_name="")
    outputs = model.get_hidden_states(images)

    # save CLS embeddings for each layer
    for l, embedding in enumerate(outputs):
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "layer_" + str(l) + ".npy")
        CLS_embedding = embedding[:,0,:] # get embedding for CLS token
        np.save(save_path, CLS_embedding)

if __name__ == "__main__":
    main() 
