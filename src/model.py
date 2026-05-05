"""Modified ResNet-18 builder + target loader.

Architecture matches tml26_task1/task_template.py exactly:
    - first conv: 3x3 instead of 7x7 (better for 32x32 inputs)
    - maxpool: removed (no aggressive early downsampling)
    - fc: 512 -> 9 classes
This is the standard "CIFAR-style" ResNet-18 adaptation.
"""
from pathlib import Path
import torch
from torchvision.models import resnet18

from .dataset import NUM_CLASSES


def build_resnet18(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """Construct an untrained modified ResNet-18 with the target architecture."""
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, num_classes)
    return model


def load_target(data_dir: Path | str, device: str = "cpu") -> torch.nn.Module:
    """Load the trained target model.pt in eval mode."""
    path = Path(data_dir) / "model.pt"
    model = build_resnet18()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
