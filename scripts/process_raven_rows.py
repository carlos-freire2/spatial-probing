import os
import numpy as np
import xml.etree.ElementTree as ET
import pickle

BASE_DIR = "/oscar/home/eshivers/cs1470_spatial_probing/spatial-probing/data/RAVEN/raven_generated"
OUT_DIR = "/oscar/home/eshivers/cs1470_spatial_probing/spatial-probing/data/raven_rows"

os.makedirs(OUT_DIR, exist_ok=True)

# all puzzle folders
configs = os.listdir(BASE_DIR)

def get_rules(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rules = []
    for rule_group in root.find("Rules"):
        for rule in rule_group:
            name = rule.attrib.get("name")
            attr = rule.attrib.get("attr")
            rules.append(f"{name}_{attr}")
    return rules  # returns list of 3 rules (one for each row)

def extract_rows(images):
    # images shape: (16, 160, 160)
    # row1: panels 0,1,2; row2: 3,4,5; row3: 6,7,8
    row1 = images[0:3]
    row2 = images[3:6]
    row3 = images[6:9]
    return [row1, row2, row3]

counter = 0
for config in configs:
    folder = os.path.join(BASE_DIR, config)
    if not os.path.isdir(folder):
        continue
    
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
    for f in files:
        npz_path = os.path.join(folder, f)
        xml_path = npz_path.replace(".npz", ".xml")
        
        # load data
        arr = np.load(npz_path)
        images = arr["image"]
        
        # extract 3 rows
        rows = extract_rows(images)
        
        # get puzzle rules
        rules = get_rules(xml_path)
        
        # save each row as one example
        for i in range(3):
            out_path = os.path.join(OUT_DIR, f"row_{counter}.pkl")
            data = {
                "images": rows[i],   # 3 images
                "rule": rules[i],    # matching rule
                "config": config
            }
            with open(out_path, "wb") as f_out:
                pickle.dump(data, f_out)
            counter += 1

print(f"Saved {counter} row examples to {OUT_DIR}")

