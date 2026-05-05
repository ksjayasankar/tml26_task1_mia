"""Dataset classes and loaders.

Mirrors the class layout used in `tml26_task1/task_template.py` so that
`torch.load` of `pub.pt` / `priv.pt` succeeds (the saved objects are
instances of these classes).

The .pt files contain:
    pub_ds.ids       list[int]            sample ids
    pub_ds.imgs      list[Tensor[3,H,W]]  RGB images, already tensors
    pub_ds.labels    list[int]            class indices (0..8)
    pub_ds.membership list[int]           1 if member of target's training set, 0 otherwise
                                          (priv_ds.membership is None for every entry)

The transform is `Resize(32) + Normalize(MEAN, STD)`. No ToTensor — the imgs are already tensors.
The Resize(32) is the upsample from 28x28 (MedMNIST native) to 32x32 (matches the modified ResNet-18).
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Normalization stats from `tml26_task1/task_template.py`.
# Values match histopathology images (predominantly pink/purple H&E staining)
# and are consistent with MedMNIST PathMNIST per-channel statistics.
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

NUM_CLASSES = 9


def default_transform():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# The pub.pt / priv.pt files were originally created with task_template.py running
# as __main__, so torch.save pickled their class references as `__main__.MembershipDataset`
# / `__main__.TaskDataset`. When we load from a different driver script, those names
# resolve to our driver, which doesn't define the classes -> AttributeError.
# Register them into __main__ on import so torch.load works regardless of caller.
import sys as _sys
_main = _sys.modules["__main__"]
if not hasattr(_main, "TaskDataset"):
    _main.TaskDataset = TaskDataset
if not hasattr(_main, "MembershipDataset"):
    _main.MembershipDataset = MembershipDataset


def load_pub(data_dir: Path | str, attach_transform: bool = True) -> MembershipDataset:
    """Load pub.pt with membership labels populated."""
    path = Path(data_dir) / "pub.pt"
    ds = torch.load(path, weights_only=False)
    if attach_transform:
        ds.transform = default_transform()
    return ds


def load_priv(data_dir: Path | str, attach_transform: bool = True) -> MembershipDataset:
    """Load priv.pt. ds.membership is a list of None."""
    path = Path(data_dir) / "priv.pt"
    ds = torch.load(path, weights_only=False)
    if attach_transform:
        ds.transform = default_transform()
    return ds
