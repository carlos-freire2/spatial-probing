import os
import re
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

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
                    print(data)
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"    # this is for apple metal gpu
    
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Generate hidden state embeddings")
    parser.add_argument("--model", choices=["DINO", "CLIP"], help="model, options are DINO or CLIP")
    parser.add_argument("--data_dir", help="path to directory where raw data files are located")
    parser.add_argument("--save_dir", help="path to directory where processed data will be saved")
    args = parser.parse_args()
    
    # we don't create OUTPUT_DIR here yet, we create subdirs dynamically but we could if we wanted
    
    # I'm sure parseargs can do this for you but I don't want to read the docs and had these handy already -_-
    model_id = DEFAULT_MODEL_ID 
    if args.model != None and args.model.upper() == DINO:
        model_id = DINO
    elif args.model.upper() == CLIP:
        model_id = CLIP

    image_dir = args.data_dir if args.data_dir != None else DEFAULT_IMAGE_DIR
    output_dir = args.save_dir if args.save_dir != None else DEFAULT_OUTPUT_DIR

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    dataset = RavenDataset(image_dir, processor)
    
    # num_workers > 0 helps prefetch data while GPU processes
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_fn, num_workers=2)

    print(f"Found {len(dataset)} RAVEN files across all subdirs")
    print("Starting extraction...")

    with torch.inference_mode():
        for batch_idx, (pixel_values, rel_paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if pixel_values is None: continue

            pixel_values = pixel_values.to(device)
            

            outputs = model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            
            all_layers = torch.stack(outputs.hidden_states)
            # only take CLS
            cls_features = all_layers[:, :, 0, :] # (Layers, Batch*16, Dim)

            # reshape back to separate files
            num_layers, _, dim = cls_features.shape
            num_files = len(rel_paths)
            panels_per_file = 16 

            reshaped_features = cls_features.view(num_layers, num_files, panels_per_file, dim)

            # SAVE LOGIC
            # since BATCH_SIZE might be > 1, we iterate through the batch but currently does nothing
            for i in range(num_files):
                current_features = reshaped_features[:, i, :, :] # (Layers, 16, Dim)
                current_rel_path = rel_paths[i] # e.g. "center_single/RAVEN_0_train.npz"
                
                # here we create the output path so we need to remove .npz extension
                file_name_no_ext = os.path.splitext(current_rel_path)[0] 
                # combine with output root: "clip_embeddings/center_single/RAVEN_0_train.pt"
                save_path = os.path.join(output_dir, f"{file_name_no_ext}.pt")
                
                # create the subdirectory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # I guess we don't really need the filename here maybe we just shove the feature?
                torch.save({
                    "features": current_features.cpu(),
                    "filename": current_rel_path
                }, save_path)

    print("---------------- Done with embedding extraction! ----------------")

if __name__ == '__main__':
    main()