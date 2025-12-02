import os
from PIL import Image
import matplotlib.pyplot as plt

RAW_DIR = "/data/ds340w/work/train_598"
OUT_DIR = "/data/ds340w/work/debug_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# take first few files from the raw folder
files = sorted(os.listdir(RAW_DIR))[:5]

print("Saving raw image snapshots from:")
for i, fname in enumerate(files):
    raw_path = os.path.join(RAW_DIR, fname)
    print("  ", raw_path)

    img = Image.open(raw_path)          # this is the TRUE raw PNG
    # if they’re grayscale, you can force it:
    # img = img.convert("L")

    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(fname)

    out_path = os.path.join(OUT_DIR, f"raw_{i:02d}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("  -> saved:", out_path)

print("Done.")
