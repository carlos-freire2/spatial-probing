"""
Extracts hidden-state embeddings for the specified model and data.
Expects data directory to have images.npy and labels.json files, 
the outputs from preprocess_RAVEN.py.
Saves embeddings as .npy files of shape (N, D), one per hidden layer.

Example: 
python scripts/extract_embeddings.py \
    --model DINOv3 \
    --data_dir /data/RAVEN/center_single/ \
    --save_dir /embdeddings/RAVEN/DINOv3/center_single/
"""

import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
import argparse
import shutil
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.dino_model import DINOv3Wrapper
from models.clip_model import CLIPWrapper
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Extract hidden state embeddings")
    parser.add_argument("--model", choices=["DINOv3", "CLIP"], help="model, options are DINOv3 or CLIP")
    parser.add_argument("--data_dir", help="path to directory where image files are located")
    parser.add_argument("--save_dir", help="path to directory where embeddings will be saved")
    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, "images.npy")
    labels_path = os.path.join(args.data_dir, "labels.json")
    images = np.load(data_path)

    # get hidden state outputs from the selected model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "DINOv3":
        model = DINOv3Wrapper(model_name="facebook/dinov3-vits16-pretrain-lvd1689m")
    if args.model == "CLIP":
        model = CLIPWrapper(model_name="openai/clip-vit-base-patch32")

    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_outputs = []
    for batch in tqdm(dataloader):
        outputs = model.get_hidden_states(batch, device)
        all_outputs.append(outputs)

    join_layers = list(zip(*all_outputs)) # group data from each layer together
    all_outputs = [torch.cat(l, dim=0) for l in join_layers]

    # save CLS embeddings for each layer
    for l, embedding in enumerate(all_outputs):
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "layer_" + str(l) + ".npy")
        CLS_embedding = embedding[:,0,:].cpu().numpy() # get embedding for CLS token
        np.save(save_path, CLS_embedding)

    labels_save_path = os.path.join(args.save_dir, "labels.json")
    shutil.copy(labels_path, labels_save_path)

class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    main() 
