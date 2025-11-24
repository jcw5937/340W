########################################
# pipeline_debugger.py                 #
# Trace images through DICOM → PNG →  #
# cropped (train_598) → augmented     #
########################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "/data/ds340w"
DATA_DIR = os.path.join(BASE_DIR, "data")
WORK_DIR = os.path.join(BASE_DIR, "work")

OUT_DIR = os.path.join(WORK_DIR, "debug_pipeline")
os.makedirs(OUT_DIR, exist_ok=True)


def save_img_array(img, out_path, cmap="gray"):
    """Save a numpy image array to disk as PNG."""
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  -> saved: {out_path}")


########################
# Stage 0: raw DICOMs  #
########################

def debug_stage0_dicom(n=3):
    print("\n=== Stage 0: DICOM → PNG ===")
    try:
        import pydicom
    except ImportError:
        print("pydicom not installed in this env; skipping Stage 0.")
        return

    # Find a few DICOM files in the manifest tree
    search_pattern = os.path.join(
        DATA_DIR, "manifest-*", "CBIS-DDSM", "*", "*", "*", "*", "*.dcm"
    )
    dicom_paths = sorted(glob.glob(search_pattern))
    if not dicom_paths:
        print("No DICOM files found with pattern:")
        print(f"  {search_pattern}")
        return

    for i, dcm_path in enumerate(dicom_paths[:n]):
        print(f"Using DICOM: {dcm_path}")
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)

        # Simple normalization to 0–1 for display
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()

        out_path = os.path.join(OUT_DIR, f"stage0_dicom_{i:02d}.png")
        save_img_array(img, out_path)


############################################
# Stage 1: converted PNGs under work/png   #
############################################

def debug_stage1_png(n=5):
    print("\n=== Stage 1: Converted PNG under work/png ===")
    search_pattern = os.path.join(WORK_DIR, "png", "**", "*.png")
    png_paths = sorted(glob.glob(search_pattern, recursive=True))
    if not png_paths:
        print("No PNGs found under /work/png. Maybe Convert_DICOM_to_PNG.py never ran, or moved them elsewhere.")
        print(f"Checked pattern: {search_pattern}")
        return

    print(f"Found {len(png_paths)} PNGs under work/png. Saving up to {n} examples.")
    for i, p in enumerate(png_paths[:n]):
        print(f"  Source: {p}")
        img = plt.imread(p)
        if img.ndim == 3:
            # convert to grayscale for consistent viewing
            img = img.mean(axis=2)
        out_path = os.path.join(OUT_DIR, f"stage1_png_{i:02d}.png")
        save_img_array(img, out_path)


########################################################
# Stage 2: cropped / ROI images (train_598)           #
########################################################

def debug_stage2_cropped(n=5):
    print("\n=== Stage 2: Cropped images under work/train_598 ===")
    cropped_dir = os.path.join(WORK_DIR, "train_598")
    if not os.path.isdir(cropped_dir):
        print(f"Directory not found: {cropped_dir}")
        return

    png_paths = sorted(
        [os.path.join(cropped_dir, f) for f in os.listdir(cropped_dir)
         if f.lower().endswith(".png")]
    )
    if not png_paths:
        print(f"No PNGs found in {cropped_dir}")
        return

    print(f"Found {len(png_paths)} PNGs in train_598. Saving up to {n} examples.")
    for i, p in enumerate(png_paths[:n]):
        print(f"  Source: {p}")
        img = plt.imread(p)
        if img.ndim == 3:
            img = img.mean(axis=2)
        out_path = os.path.join(OUT_DIR, f"stage2_cropped_{i:02d}.png")
        save_img_array(img, out_path)


########################################################
# Stage 3: augmented images (train_598_augmented)      #
########################################################

def debug_stage3_augmented(n=5):
    print("\n=== Stage 3: Augmented images under work/train_598_augmented ===")
    aug_dir = os.path.join(WORK_DIR, "train_598_augmented")
    if not os.path.isdir(aug_dir):
        print(f"Directory not found: {aug_dir}")
        return

    png_paths = sorted(
        [os.path.join(aug_dir, f) for f in os.listdir(aug_dir)
         if f.lower().endswith(".png")]
    )
    if not png_paths:
        print(f"No PNGs found in {aug_dir}")
        return

    print(f"Found {len(png_paths)} PNGs in train_598_augmented. Saving up to {n} examples.")
    for i, p in enumerate(png_paths[:n]):
        print(f"  Source: {p}")
        img = plt.imread(p)
        if img.ndim == 3:
            img = img.mean(axis=2)
        out_path = os.path.join(OUT_DIR, f"stage3_aug_{i:02d}.png")
        save_img_array(img, out_path)


if __name__ == "__main__":
    print("Output directory:", OUT_DIR)
    debug_stage0_dicom(n=3)
    debug_stage1_png(n=5)
    debug_stage2_cropped(n=5)
    debug_stage3_augmented(n=5)
    print("\n=== Done. Check the images in debug_pipeline/ ===")
