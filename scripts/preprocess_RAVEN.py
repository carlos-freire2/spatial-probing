"""
Preprocesses RAVEN dataset by stitching puzzle images into a grid.

Example: 
python scripts/preprocess_RAVEN.py \
    --data_dir /data/RAVEN-10000/ \
    --save_dir /data/RAVEN_preprocessed/ \
    --blank
"""

import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

def main():
    parser = argparse.ArgumentParser(description="Preprocess RAVEN dataset")
    parser.add_argument("--data_dir", help="path to directory where raw data files are located")
    parser.add_argument("--save_dir", help="path to directory where processed data will be saved")
    parser.add_argument("--blank", action='store_true', help="optional, leaves out target image from puzzle")
    args = parser.parse_args()

    dirs = [d.name for d in os.scandir(args.data_dir) if d.is_dir()]

    for type_dir in dirs:

        images = []
        labels = {}

        type_dir_path = os.path.join(args.data_dir, type_dir)
        filenames = os.listdir(type_dir_path)

        # sort filenames so that indices will be in order
        def get_num(name):
            return int(name[name.find("_")+1:name.rfind("_")])
        filenames = sorted(filenames, key=get_num, reverse=False)

        for i, f in tqdm(enumerate(filenames), total=len(filenames)):
            if f.endswith(".xml"): continue

            # check for unexpected misalignment of labels
            idx_first_underscore = f.find("_")
            idx_last_underscore = f.rfind("_")
            idx_last_period = f.rfind(".") 
            num = f[idx_first_underscore+1:idx_last_underscore]
            if (len(images)) != int(num): raise RuntimeError("label indices misaligned")

            # get stitched image
            npz_path = os.path.join(args.data_dir, type_dir, f)
            data = np.load(npz_path)
            grid = stitch_grid(data["image"], data["target"], blank=args.blank)
            images.append(grid)

            # get rule data
            split = f[idx_last_underscore+1:idx_last_period]

            xml_data_filename = f[:idx_last_period] + ".xml"
            xml_path = os.path.join(args.data_dir, type_dir, xml_data_filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            rules_data = get_rules(root)

            labels[num] = {
                "split": split,
                "rules": rules_data,
                "filename": f
            }

        save_path = os.path.join(args.save_dir, type_dir)
        os.makedirs(save_path, exist_ok=True)

        images = np.array(images)
        image_save_path = os.path.join(args.save_dir, type_dir, "images.npy")
        np.save(image_save_path, images)

        labels_save_path = os.path.join(args.save_dir, type_dir, "labels.json")
        with open(labels_save_path, "w") as f:
            json.dump(labels, f, indent=4)
        
def get_rules(root):
    rules_element = root.find("Rules")
    rules_data = {}
    for rule_group in rules_element.findall("Rule_Group"):
        r = {}
        rule_count = 0
        for rule in rule_group.findall("Rule"):
            r["rule_" + str(rule_count)] = {
                "name": rule.get("name"),
                "attribute": rule.get("attr")
            }
            rule_count += 1

        rules_data["rule_group_" + rule_group.get("id")] = r
    return rules_data

def stitch_grid(images, target, blank=False):
    if len(images) == 0:
        print("No images to stitch")
        raise RuntimeError()
    
    dtype = images[0].dtype
    grid = []
    for i in range(3):
        row = []
        for j in range(3):
            if (i == 2) and (j == 2): 
                if blank == True:
                    # make sure we are not converting to float64s here
                    # model preprocessor will handle all of that
                    blank = np.ones(images[0].shape, dtype=dtype)*255
                    blank = np.pad(blank, 6, constant_values=255)
                    row += [blank]
                else:
                    # use target image
                    im = np.pad(images[8 + target], 2)
                    im = np.pad(im, 4, constant_values=255)
                    row += [im]
                continue
            im = np.pad(images[i*3+j], 2)
            im = np.pad(im, 4, constant_values=255)
            row += [im]
        row = np.concatenate(row, axis=1)
        grid += [row]
    grid = np.concatenate(grid, axis=0)
    grid = np.pad(grid, 2, mode='constant', constant_values=255)
    return grid.astype(np.uint8) # jic

if __name__ == "__main__":
    main() 