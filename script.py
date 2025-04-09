import os
import shutil
import numpy as np
from scipy.io import loadmat

# Paths
VAL_DIR = "/mnt/data/imagenet/val"
META_MAT_PATH = os.path.join(VAL_DIR, "meta.mat")
GT_FILE_PATH = os.path.join(VAL_DIR, "ILSVRC2012_validation_ground_truth.txt")

# Load meta.mat
meta = loadmat(META_MAT_PATH, squeeze_me=True)
synsets_struct = meta['synsets']

# Extract synsets where ILSVRC2012_ID <= 1000
synsets = []
for item in synsets_struct:
    if int(item[0]) <= 1000:  # ILSVRC2012_ID
        wnid = str(item[1])   # WNID
        synsets.append(wnid)

# Read ground truth labels (1-based)
with open(GT_FILE_PATH, "r") as f:
    gt_labels = [int(line.strip()) for line in f.readlines()]

# Move each image to its correct folder
for i, label in enumerate(gt_labels):
    synset = synsets[label - 1]  # convert 1-based label to 0-based index
    filename = f"ILSVRC2012_val_{i+1:08d}.JPEG"
    src_path = os.path.join(VAL_DIR, filename)
    dst_dir = os.path.join(VAL_DIR, synset)
    dst_path = os.path.join(dst_dir, filename)

    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(src_path, dst_path)

print("✅ All validation images successfully organized into synset folders.")
