import os
from PIL import Image

BASE_DIR = "/data/ds340w"
WORK_DIR = os.path.join(BASE_DIR, "work")

IN_DIR = os.path.join(WORK_DIR, "full_train_needed")
OUT_DIR = os.path.join(WORK_DIR, "train_same_size")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SIZE = (800, 800)

for filename in os.listdir(IN_DIR):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(IN_DIR, filename))
        resized = img.resize(TARGET_SIZE)
        resized.save(os.path.join(OUT_DIR, filename))
        print("Converted:", filename)
