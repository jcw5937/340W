import os
import shutil
import re

##################################################
# This file will fix an error made in naming the #
# earlier documents, to match the model eval .py #
##################################################

BASE_DIR = "/data/ds340w"
WORK_DIR = os.path.join(BASE_DIR, "work")

BAD_DIR = os.path.join(WORK_DIR, "train_598_augmented")
FIXED_DIR = os.path.join(WORK_DIR, "train_598_augmented_fixed")

os.makedirs(FIXED_DIR, exist_ok=True)

# Regex pattern to extract the real filename from your current mangled names
# Example input:
#   _data_ds340w_work_png_Mass-Training_P_01796_LEFT_CC_1__1-1_3.png
#
# It extracts:
#   Mass-Training_P_01796_LEFT_CC_1__1-1
pattern = re.compile(r".*?(Mass-(Training|Test)_P_\d+_[A-Z]+_(CC|MLO).*?)(?:_\d+)?\.png$")

for fname in os.listdir(BAD_DIR):
    if not fname.endswith(".png"):
        continue

    full_path = os.path.join(BAD_DIR, fname)

    match = pattern.match(fname)
    if not match:
        print(f"WARNING: couldn't parse {fname}")
        continue

    base_name = match.group(1)

    # If the filename ends with "_X.png", preserve the augmentation index
    augmentation_match = re.search(r"_(\d+)\.png$", fname)
    if augmentation_match:
        idx = augmentation_match.group(1)
        new_name = f"{base_name}_{idx}.png"
    else:
        new_name = f"{base_name}.png"

    dest_path = os.path.join(FIXED_DIR, new_name)

    shutil.copy2(full_path, dest_path)

    print(f"{fname}  -->  {new_name}")

print("Done! All repaired files saved in train_598_augmented_fixed")
