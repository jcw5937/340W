######################
# Image augmentation #
######################

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#######################
# Updated for Cluster #
#######################

BASE_DIR = "/data/ds340w"
WORK_DIR = os.path.join(BASE_DIR, "work")

# Input: cropped ROI patches
original_dir  = os.path.join(WORK_DIR, "train_598")
# Output: augmented patches (NEW folder name to avoid mixing with old files)
augmented_dir = os.path.join(WORK_DIR, "train_598_aug_v2")
os.makedirs(augmented_dir, exist_ok=True)

# ImageDataGenerator for augmentation (same as original)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# List all original image files
original_files = sorted([
    f for f in os.listdir(original_dir)
    if f.lower().endswith(".png")
])

def save_augmented_images(img, prefix, idx):
    filename = os.path.join(augmented_dir, f"{prefix}_{idx}.png")
    cv2.imwrite(filename, img)

for filename in original_files:
    in_path = os.path.join(original_dir, filename)
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Warning: could not read {in_path}")
        continue

    # Make sure we are working with a single channel (grayscale)
    if len(img.shape) == 3:
        # convert BGR -> GRAY if needed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float32 in [0, 1]
    img = img.astype(np.float32)
    max_val = np.max(img)
    if max_val > 0:
        img = img / max_val
    else:
        # completely black image – skip
        print(f"Warning: zero max in {in_path}, skipping.")
        continue

    # Expand dims: (H, W) -> (1, H, W, 1)
    img = np.expand_dims(img, axis=(0, -1))

    # Generate augmented images
    i = 0
    for batch in datagen.flow(img, batch_size=1):
        # batch[0] is (H, W, 1) in [0, 1]
        aug = batch[0]

        # Convert to 8-bit [0,255] for stable saving/viewing
        aug_uint8 = (aug * 255.0).clip(0, 255).astype(np.uint8)

        prefix = os.path.splitext(filename)[0]
        save_augmented_images(aug_uint8, prefix, i + 1)
        print(f"Augmented image {i + 1} saved from: {filename}")

        i += 1
        if i >= 5:  # match original: 5 augmented images per input
            break

print("Data augmentation completed.")
