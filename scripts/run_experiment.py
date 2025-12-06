"""
Runs a single probe experiment for the specified conditions.
Saves probe.pt, the trained probe weights, and metrics.json, the final metrics 
and model configs, in the directory defined by save_dir/model/layer/probe/dataset_name/data_config/task/. 

Example: 
python scripts/run_experiment.py \
    --embedding_dir embeddings/RAVEN/DINOv3/ \
    --save_dir experiments/ \
    --model DINOv3 \
    --layer 1 \
    --probe linear \
    --dataset_name RAVEN \
    --data_config center_single \
    --task Type \
    --seed 42 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 10 
"""

import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
import argparse
import json
import numpy as np
import torch
from probing.linear_probe import LinearProbe
from probing.mlp_probe import MLPProbe
from probing.train import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(description="Run probe experiement")
    parser.add_argument("--embedding_dir", help="")
    parser.add_argument("--save_dir", help="")
    parser.add_argument("--model", help="model name")
    parser.add_argument("--layer", help="")
    parser.add_argument("--probe", choices=["linear", "MLP"], help="probe model type, options are linear or MLP")
    parser.add_argument("--dataset_name", choices=["RAVEN"], help="name of dataset, option is RAVEN")
    parser.add_argument("--data_config", choices=["center_single"], help="")
    parser.add_argument("--task", help="")
    parser.add_argument("--seed", default=42, help="")
    parser.add_argument("--lr", default=0.001, help="")
    parser.add_argument("--batch_size", default=32, help="")
    parser.add_argument("--epochs", default=10, help="")
    args = parser.parse_args()

    # load embeddings and labels
    data_path = os.path.join(args.embedding_dir, args.data_config, "layer_" + str(args.layer) + ".npy")
    data = np.load(data_path)

    labels_path = os.path.join(args.embedding_dir, args.data_config, "labels.json")
    with open(labels_path, "r") as f:
        labels = json.load(f)

    if args.dataset_name == "RAVEN":
        config_task_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAVEN_config_tasks.json")
        with open(config_task_path, "r") as f:
            task_json = json.load(f)
        train_data, train_labels, val_data, val_labels, test_data, test_labels, class_labels = get_train_val_test(data, labels, args.task, args.data_config, task_json)

    # initialize and train probe
    set_seed(int(args.seed))
    num_classes = len(class_labels.keys())
    if args.probe == "linear":
        probe = LinearProbe(num_classes=num_classes)
    if args.probe == "MLP":
        probe = MLPProbe(num_classes=num_classes)

    metrics, trained_model = train_and_evaluate(
        model=probe, 
        train_data=train_data, 
        train_labels=train_labels, 
        val_data=val_data, 
        val_labels=val_labels,
        test_data=test_data, 
        test_labels=test_labels,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs)
    
    # save trained model weights
    save_path = os.path.join(args.save_dir, args.model, args.layer, args.probe, args.dataset_name, args.data_config, args.task)
    checkpoint_save_path = os.path.join(save_path, "model.pth")
    torch.save(trained_model.state_dict(), checkpoint_save_path)

    # save model metrics and configs
    metrics["configs"] = {
        "embedding_dir": args.embedding_dir,
        "save_dir": args.save_dir,
        "model": args.model,
        "layer": args.layer,
        "probe": args.probe,
        "dataset_name": args.dataset_name,
        "data_config": args.data_config,
        "task": args.task,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs
    }
    metrics["classes"] = class_labels
    metrics_save_path = os.path.join(save_path, "metrics_configs.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)

def get_train_val_test(data, labels, task, data_config, task_json):
    classes = task_json[data_config][task]

    # split into train, validation, and test sets
    train_data = []
    val_data = []
    test_data = []

    train_labels = []
    val_labels = []
    test_labels = []
    for k in labels.keys():
        rules = labels[k]["rules"]
        rules_lookup_0, rules_lookup_1 = get_rules_lookup(labels, k, rules)

        if labels[k]["split"] == "train": 
            train_data.append(data[int(k)])
            if task.endswith("1"):
                rule = rules_lookup_1[task]
                train_labels.append(classes.index(rule))
            else: 
                rule = rules_lookup_0[task]
                train_labels.append(classes.index(rule))

        if labels[k]["split"] == "val": 
            val_data.append(data[int(k)])
            if task.endswith("1"):
                rule = rules_lookup_1[task]
                val_labels.append(classes.index(rule))
            else: 
                rule = rules_lookup_0[task]
                val_labels.append(classes.index(rule))

        if labels[k]["split"] == "test": 
            test_data.append(data[int(k)])
            if task.endswith("1"):
                rule = rules_lookup_1[task]
                test_labels.append(classes.index(rule))
            else: 
                rule = rules_lookup_0[task]
                test_labels.append(classes.index(rule))
    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    class_labels = {i:t for i,t in enumerate(classes)}

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, class_labels

def get_rules_lookup(labels, k, rules):
    rules_lookup_0 = {}
    rules_lookup_1 = {}

    for i in range(len(rules["rule_group_0"])):
        r = rules["rule_group_0"]["rule_" + str(i)]
        rules_lookup_0[r["attribute"]] = r["name"]

    if len(labels[k]["rules"]) > 1: 
        rules_lookup_1 = {}
        for i in range(len(rules["rule_group_1"])):
            r = rules["rule_group_1"]["rule_" + str(i)]
            rules_lookup_1[r["attribute"]] = r["name"]
            
    return rules_lookup_0, rules_lookup_1


if __name__ == "__main__":
    main() 
