import os
import pydicom
import matplotlib.pyplot as plt

# adjust this once you see what's inside manifest-...
ROOT = "/data/ds340w/data/manifest-ZkhPvrLo5216730872708713142"

# find one DICOM file
dcm_path = None
for dirpath, dirnames, filenames in os.walk(ROOT):
    for f in filenames:
        if f.lower().endswith(".dcm"):
            dcm_path = os.path.join(dirpath, f)
            break
    if dcm_path is not None:
        break

print("Using DICOM:", dcm_path)

ds = pydicom.dcmread(dcm_path)
img = ds.pixel_array

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Raw DICOM example")
plt.savefig("/data/ds340w/work/debug_raw/dicom_example.png", dpi=200)
plt.close()
