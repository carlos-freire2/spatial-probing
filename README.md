# Spatial Relation Probing in CLIP and DINO

This project investigates whether pretrained vision models encode spatial relationship information such as *on*, *under*, *behind*, and *in front of*. We compare two major vision foundation models:

- **CLIP (Contrastive Language–Image Pretraining)**  
  A model trained to align images and text descriptions. It may learn spatial concepts indirectly from captions (e.g., “cat on chair”, “man behind horse”).

- **DINO (Self-Supervised Vision Transformer)**  
  A model trained without labels that learns strong geometric and object-centric representations through self-supervised consistency.

By using probing, we test whether these models’ internal embeddings contain spatial reasoning abilities before any fine-tuning.

---

## What Is Probing?

A **probe** is a small classifier (often a simple linear layer) trained on top of **frozen** neural network embeddings.

- The **backbone model (CLIP or DINO) is kept unchanged**.  
- Only the probe learns to map embeddings to spatial relation labels.
- Probing is done **layer-wise** to see how information evolves across transformer layers.

Probing allows us to measure what information already exists in pretrained models without modifying them.

---

## Dataset

We use **two datasets** that provide annotated spatial relationships between object pairs:

### **1. Visual Relationship Detection (VRD) Dataset**
Contains:
- ~5000 images  
- ~38,000 visual relationships  
- 100 object categories  
- 70 predicate categories (including spatial relations such as *on*, *under*, *in front of*, *behind*)

### **2. SpatialSense Dataset**
A challenging benchmark created to evaluate fine-grained spatial relation understanding.  
It contains:
- 11,569 images  
- 17,498 human-verified relations  
- Balanced positive/negative examples for each spatial predicate  
- Relations designed to avoid simple 2D or language shortcuts

SpatialSense is included to provide a more reliable measure of true spatial reasoning.

---

## Installation

```
git clone <this_repo_url>
cd spatial-probing
pip install -r requirements.txt
```

To generate a dataset
```
cd data/RAVEN
python src/dataset/main.py --num-samples <number of samples per configuration> --save-dir <directory to save the dataset>
```
This will create several new folders with numpy archive and xml data which we can use. Until we have a dataset we are certain we want to use, I'd avoid pushing them up to the central repo (will cause merge conflicts for everyone).

---

## Team

Erica Shivers
Jahan Khan
Madeleine Fenner
Carlos Freire


---

## References

- Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP)  
- Caron et al., “Emerging Properties in Self-Supervised Vision Transformers” (DINO)  
- Lu et al., “Visual Relationship Detection with Language Priors” (VRD)  
- Yang et al., “SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition”
