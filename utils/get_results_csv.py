'''
Example:
python utils/get_results_csv.py \
    --data_dir experiments/
    --save_dir results/
'''

import os
import json
import csv
import argparse

results_path = "experiments/"

parser = argparse.ArgumentParser(description="Write experiment results to csv")
parser.add_argument("--data_dir", help="path to root directory where experiments are located")
parser.add_argument("--save_dir", help="path to directory where csv will be saved")
args = parser.parse_args()

header = ["model", "layer", "probe", "dataset_name", "data_config", "task", "acc"]
def metrics_search(dir):
    paths = []

    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            paths.extend(metrics_search(full_path))
        elif f.endswith('.json'):
            paths.append(full_path)

    return paths

if args.data_dir:
    results_path = args.data_dir
else:
    results_path = results_path

metrics_paths = metrics_search(results_path)

rows = []
rows.append(header)
for m in metrics_paths:

    dirs = m.split("/")
    if results_path.endswith("/"): results_path = results_path[:-1]
    results_dir = results_path[results_path.rfind("/")+1:]
    
    labels = dirs[dirs.index(results_dir)+1:-1]

    with open(m, 'r', encoding='utf-8') as f:
        m_json = json.load(f)
    labels.append(m_json['test_accuracy'])
    rows.append(labels)

if args.save_dir:
    csv_save_path = os.path.join(args.save_dir, "results.csv")
else:
    print(os.path.dirname(os.path.abspath(__file__)))
    csv_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

with open(csv_save_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
