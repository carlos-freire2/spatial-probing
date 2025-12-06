import os
import re
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from models.clip_model import CLIPWrapper
from models.dino_model import DINOv3Wrapper

# --- CONFIG ---
DEFAULT_IMAGE_DIR = "./RAVEN-10000"                 # Root folder containing subdirs (center_single, etc.)
DEFAULT_OUTPUT_DIR = "./clip_embeddings_whole"      # Root folder for output
BATCH_SIZE = 1                                      # Keep at 1 unless we decide a new file -> embedding structure
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"   # This can stay hardcoded
DINO_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
CLIP_ID= "openai/clip-vit-base-patch32"
CLIP = "CLIP"
DINO = "DINO"

class RavenDataset(Dataset):
    def __init__(self, img_dir, processor):
        self.img_dir = img_dir
        self.processor = processor
        self.image_files = []
        
        # walk through all directories starting from img_dir
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.endswith(('.npz', '.npy')): # you guys can decide which we save as, but I'll leave this here in case
                    full_path = os.path.join(root, file)
                    # create the relative path (e.g., "center_single/RAVEN_0_train.npz")
                    rel_path = os.path.relpath(full_path, img_dir)
                    self.image_files.append(rel_path)
        
        # sort to ensure consistent processing order (otherwise we _1_, _10_, _100_ instead of _1_, _2_, _3_)
        self.image_files.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', f)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # rel_path e.g., "center_single/RAVEN_0_train.npz"
        rel_path = self.image_files[idx]
        file_path = os.path.join(self.img_dir, rel_path)
        
        try:
            if file_path.endswith('.npz'): # npz has multiple arrays
                with np.load(file_path) as data:
                    # print(data)
                    # look for 'image' key, throw error otherwise (data must be corrupted)
                    if 'image' in data.files:
                        # keys: target, predict, image, structure, meta_matrix, meta_structure, meta_target
                        img_array = data['image']
                    else:
                        raise Exception() # we always have image key

            else:
                img_array = np.load(file_path)  

            # shape (16, 160, 160)
            if img_array.ndim == 3:
                num_panels = img_array.shape[0] # Should be 16
                
                panel_tensors = []
                for i in range(num_panels):
                    panel = img_array[i]
                    pil_img = Image.fromarray(panel).convert("RGB")
                    inputs = self.processor(images=pil_img, return_tensors="pt")
                    panel_tensors.append(inputs['pixel_values'].squeeze(0))
                
                # shape (16, 3, 224, 224) for RGB
                return torch.stack(panel_tensors), rel_path

            else:
                print(f"Skipping {rel_path}: Unexpected shape {img_array.shape}")
                return torch.zeros(16, 3, 224, 224), "error"

        except Exception as e:
            print(f"Error loading {rel_path}: {e}")
            return torch.zeros(16, 3, 224, 224), "error"

# NOTE: If we ever want to batch these instead of one per file we can use this.
# DataLoader by default adds a dimension so we either need to pass this in or call squeeze
# in the loop (which seems unnecessary).
def collate_fn(batch):
    tensors, filenames = zip(*batch)
    
    valid_tensors = []
    valid_names = []
    for t, f in zip(tensors, filenames):
        if f != "error":
            valid_tensors.append(t)
            valid_names.append(f)
        else: 
            # this has never fired but we can save to a file in case?
            print(f"Skipping error file in batch: {f}") 

    if not valid_tensors:
        return None, None

    # Stack (DataLoader adds the batch automatically): (Batch, 16, 3, 224, 224) -> reshape -> (Batch*16, 3, 224, 224)
    batched_tensors = torch.stack(valid_tensors)
    b, panels, c, h, w = batched_tensors.shape
    flattened_tensors = batched_tensors.view(b * panels, c, h, w)
    
    return flattened_tensors, valid_names

def stitch_grid(images, blank=False):
    grid = []
    for i in range(3):
        row = []
        for j in range(3):
            if (i == 2) & (j == 2) & (blank == True): 
                blank = np.ones(images[0].shape)*255
                blank = np.pad(blank, 6, constant_values=255)
                row += [blank]
                continue
            im = np.pad(images[i*3+j], 2)
            im = np.pad(im, 4, constant_values=255)
            row += [im]
        row = np.concatenate(row, axis=1)
        grid += [row]
    grid = np.concatenate(grid, axis=0)
    grid = np.pad(grid, 2, mode='constant', constant_values=255)
    return grid

def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from RAVEN dataset")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_IMAGE_DIR, 
                        help="Root folder containing .npz/.npy files")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help="Folder to save .pt embeddings")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, 
                        help="HuggingFace Model ID (e.g., openai/clip-vit-base-patch32)")
    return parser.parse_args()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"    # this is for apple metal gpu
    
    print(f"Using device: {device}")

    args = parse_args()
    
    # we don't create OUTPUT_DIR here yet, we create subdirs dynamically but we could if we wanted
        
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data:  {args.data_dir}")
    print(f"  Out:   {args.save_dir}")
    
    # I'm sure parseargs can do this for you but I don't want to read the docs and had these handy already -_-
    if args.model.upper() == DINO:
        ## this won't actually work right now but feel free to add whatever / change things 
        ## and it doesn't actually matter how modular this is if they aren't fitting perfectly
        ## just rip them apart
        model_id = DINO_ID
        model = DINOv3Wrapper(model_id)
    elif args.model.upper() == CLIP:
        model_id = CLIP_ID
        model = CLIPWrapper(model_id)
    else:
        print("couldn't find model")
        return

    image_dir = args.data_dir if args.data_dir != None else DEFAULT_IMAGE_DIR
    output_dir = args.save_dir if args.save_dir != None else DEFAULT_OUTPUT_DIR

    processor = model.processor
    model = model.to(device) # move to gpu (no-op if running on cpu)
    model.eval()

    dataset = RavenDataset(image_dir, processor)
    
    # num_workers > 0 helps prefetch data while GPU processes
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_fn, num_workers=2)

    print(f"Found {len(dataset)} RAVEN files across all subdirs")
    print("Starting extraction...")

    # 13 layers (1 embeddings + 12 encoder blocks)
    num_layers = 13 
    
    # list of lists to hold numpy arrays for each layer (if it's slow we can optimize this)
    layer_storage = [[] for _ in range(num_layers)]
    all_filenames = []

    with torch.inference_mode():
        for batch_idx, (pixel_values, rel_paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if pixel_values is None: continue

            pixel_values = pixel_values.to(device)
            
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)

            # shape (Batch, Seq_Len, Dim) -> (Batch, 197 ## this is 1 cls + 196 patches ##, 768)
            for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
                # extract CLS token: (Batch, Seq, Dim) -> (Batch, Dim)
                cls_token = layer_tensor[:, 0, :]
                
                # move to CPU and store as numpy
                layer_storage[layer_idx].append(cls_token.cpu().numpy())
            
            # Store filenames
            all_filenames.extend(rel_paths)
            
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "filenames.npy"), np.array(all_filenames))

    # NOTE: JIC we need to double check which indexes refer to which files (we are sorting so it should be fine) 
    print(f"Saved filenames.npy ({len(all_filenames)} files)")

    # SAVE LOGIC 
    for layer_idx in range(num_layers):
        # concat all batches for this layer
        full_layer_array = np.concatenate(layer_storage[layer_idx], axis=0) # shape: (Total_Images, Dim)
        
        save_name = f"layer_{layer_idx}.npy"
        save_path = os.path.join(output_dir, save_name)
        
        np.save(save_path, full_layer_array)
        # just a sanity check can comment out
        print(f"saved file {save_name} | shape: {full_layer_array.shape}") 

    print("---------------- Done with embedding extraction! ----------------")

if __name__ == '__main__':
    main()