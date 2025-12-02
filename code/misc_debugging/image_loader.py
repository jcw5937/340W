import os
import pydicom
import matplotlib.pyplot as plt

ROOT = "/data/ds340w/data/manifest-ZkhPvrLo5216730872708713142"

# find first DICOM
dcm_path = None
for dirpath, dirnames, filenames in os.walk(ROOT):
    for f in filenames:
        if f.lower().endswith(".dcm"):
            dcm_path = os.path.join(dirpath, f)
            break
    if dcm_path is not None:
        break

print("Using DICOM:", dcm_path)

# load dicom
ds = pydicom.dcmread(dcm_path)
img = ds.pixel_array

# save PNG version so you can inspect it
out_path = "/data/ds340w/work/debug_raw/dicom_example.png"

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.savefig(out_path, dpi=200)
print("Saved:", out_path)
