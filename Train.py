import torch
import numpy as np
import random
from ultralytics import YOLO

def set_seed(seed=42):
    """Set random seeds for reproducible initialization"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (optional, may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed before any model operations
set_seed(42)


model = YOLO("ultralytics/cfg/models/11/yolo11l-cls.yaml").load("yolo11l-cls.pt")


# Set seed again before training to ensure reproducible training behavior
set_seed(42)

# Train with explicit pretrained flag
results = model.train(
    data="path/to/your/dataset",
    epochs=100,
    imgsz=224,
    batch=16,
    device=0,
    project="test_yolo11_project",
    name="test_architecture",
    half=False,
    amp=False,
    pretrained=True,
    seed=42
)

# Validate the best checkpoint from this run on the test split
set_seed(42)
metrics = model.val(split='test')
