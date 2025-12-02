####################################################
# Calculate cropped area and full image size        #
# Cluster-ready version                             #
####################################################

import os
import re
import pandas as pd
from PIL import Image
import numpy as np

BASE_DIR = "/data/ds340w"
WORK_DIR = os.path.join(BASE_DIR, "work")

####################################################
# Helper: extract Mass/Calc, P number, LEFT/RIGHT, CC/MLO
# Regex ignores extra suffixes like _1, _1__1-2, etc.
####################################################

def extract_groups(image_path, mode="Test"):
    pattern = rf"(Mass|Calc)-{mode}_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)"
    match = re.search(pattern, image_path)
    if not match:
        return None

    mass_or_calc = match.group(1)
    number = match.group(2)
    side = match.group(3)
    view = match.group(4)

    return f"{mass_or_calc}-{mode}_P_{number}_{side}_{view}"

####################################################
# Calculate non-zero area
####################################################

def compute_nonzero_area(image_path):
    try:
        image = Image.open(image_path)
        image_array = np.array(image)

        nonzero_pixels = np.count_nonzero(image_array)
        total_pixels = image_array.size
        percentage = (nonzero_pixels / total_pixels) * 100

        return percentage
    except:
        return None

####################################################
# PROCESS TEST IMAGES
####################################################

roi_folder = os.path.join(WORK_DIR, "roi_train_needed")
df_test = pd.DataFrame(columns=["name", "file_path", "area_percentage"])

for filename in os.listdir(roi_folder):
    if filename.endswith(".png"):

        image_path = os.path.join(roi_folder, filename)

        name = extract_groups(image_path, mode="Test")
        if name is None:
            continue

        pct = compute_nonzero_area(image_path)
        df_test.loc[len(df_test)] = [name, image_path, pct]

print(f"[OK] Test images processed: {len(df_test)}")

####################################################
# PROCESS TRAINING IMAGES
####################################################

df_train = pd.DataFrame(columns=["name", "file_path", "area_percentage"])

for filename in os.listdir(roi_folder):
    if filename.endswith(".png"):

        image_path = os.path.join(roi_folder, filename)

        name = extract_groups(image_path, mode="Training")
        if name is None:
            continue

        pct = compute_nonzero_area(image_path)
        df_train.loc[len(df_train)] = [name, image_path, pct]

print(f"[OK] Training images processed: {len(df_train)}")

####################################################
# MERGE + ADD PATHOLOGY
####################################################

merged = pd.concat([df_train, df_test]).sort_values("name").reset_index(drop=True)

pathology = pd.read_csv(os.path.join(WORK_DIR, "all_mass_pathology.csv"))

def lookup_pathology(name):
    for full in pathology["full_name"]:
        if full in name:
            return pathology.loc[pathology["full_name"] == full, "pathology"].values[0]
    return None

merged["pathology"] = merged["name"].apply(lookup_pathology)

out1 = os.path.join(WORK_DIR, "598_percentage_all.csv")
merged.to_csv(out1, index=False)

print(f"[DONE] Saved {out1} with {len(merged)} rows")

####################################################
# FULL IMAGE SIZE
####################################################

full_folder = os.path.join(WORK_DIR, "full_train_needed")

subject_ids = []
widths = []
heights = []

for f in os.listdir(full_folder):
    if f.endswith(".png"):
        path = os.path.join(full_folder, f)
        subj = f.split("full_")[1].split(".png")[0]

        img = Image.open(path)
        w, h = img.size

        subject_ids.append(subj)
        widths.append(w)
        heights.append(h)

df_full = pd.DataFrame({
    "Subject_ID": subject_ids,
    "Width": widths,
    "Height": heights
})

df_full["pathology"] = df_full["Subject_ID"].apply(lookup_pathology)

out2 = os.path.join(WORK_DIR, "heaght_width_FULL.csv")
df_full.to_csv(out2, index=False)

print(f"[DONE] Saved {out2} with {len(df_full)} rows")
