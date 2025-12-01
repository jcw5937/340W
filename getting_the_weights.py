import torch
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

# Triggers download of the EXACT weights behind ResNet50_Weights.DEFAULT
model = resnet50(weights=ResNet50_Weights.DEFAULT)

cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
print("Weights cached in:", cache_dir.resolve())
for p in cache_dir.glob("resnet50*.pth"):
    print("Found:", p.name)
