"""
Functions and classes for returning image datasets
"""

from typing import Tuple

import torch as T
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import Dataset


def compose_transforms(
    to_tens: bool = True,
    resize: int = 64,
    rand_rotate: int = 0,
    rand_crop: int = 0,
    center_crop: int = 0,
    augment: bool = False,
    shift_scale: tuple = (0, 1),
) -> Tuple[tv.transforms.Compose, tv.transforms.Compose]:
    """Returns a composed list of torchvision transforms for image processing
    for the forward and the backward process"""

    fwd_trans = []
    bck_trans = []
    if augment:
        fwd_trans.append(
            tv.transforms.AutoAugment(tv.transforms.AutoAugmentPolicy.IMAGENET)
        )
    if to_tens:
        fwd_trans.append(tv.transforms.ToTensor())
    if resize:
        fwd_trans.append(tv.transforms.Resize(resize))
    if rand_rotate:
        fwd_trans.append(tv.transforms.RandomRotation(rand_rotate))
    if rand_crop:
        fwd_trans.append(tv.transforms.RandomCrop(rand_crop))
    if center_crop:
        fwd_trans.append(tv.transforms.CenterCrop(center_crop))
    if shift_scale != (0, 1):
        fwd_trans.append(tv.transforms.Normalize(*shift_scale))
        bck_trans.append(
            tv.transforms.Normalize(
                -shift_scale[0] / shift_scale[1], 1 / shift_scale[1]
            )
        )

    return tv.transforms.Compose(fwd_trans), tv.transforms.Compose(bck_trans)


def load_image_dataset(name: str, transforms: tv.transforms, is_train: bool = True):
    """Returns an image train and validation dataset after applying some transforms"""

    if name == "cifar10":
        dataset = tv.datasets.CIFAR10(
            root="data", train=is_train, download=True, transform=transforms
        )
    elif name == "mnist":
        dataset = tv.datasets.MNIST(
            root="data", train=is_train, download=True, transform=transforms
        )
    elif name == "imagenet":
        dataset = tv.datasets.ImageNet(
            root="data",
            split="train" if is_train else "val",
            download=True,
            transform=transforms,
        )
    elif name == "celeba":
        dataset = tv.datasets.CelebA(
            root="data",
            split="train" if is_train else "valid",
            download=True,
            transform=transforms,
        )
    else:
        raise ValueError(f"Unknown image dataset name: {name}")

    return dataset


class ClassificationImageDataset(Dataset):
    """A wrapper for the image datasets so certain attributes can be saved"""

    def __init__(self, is_train: bool, name: str, trans_kwargs: dict) -> None:
        super().__init__()
        self.name = name
        self.is_train = is_train
        self.fwd_trans, self.bck_trans = compose_transforms(**trans_kwargs)
        self.dataset = load_image_dataset(name, self.fwd_trans, is_train)

    def __getitem__(self, idx) -> tuple:
        image, label = self.dataset[idx]
        ctxt = F.one_hot(T.tensor(label, dtype=T.long), num_classes=10)
        return image, ctxt, label

    def __len__(self) -> int:
        return len(self.dataset)

    def image_shape(self) -> list:
        return self.dataset[0][0].shape

    def num_classes(self) -> list:
        return len(self.dataset.classes)


class ImageDataset(Dataset):
    """A wrapper for the image datasets so certain attributes can be saved"""

    def __init__(self, is_train: bool, name: str, trans_kwargs: dict) -> None:
        super().__init__()
        self.name = name
        self.is_train = is_train
        self.fwd_trans, self.bck_trans = compose_transforms(**trans_kwargs)
        self.dataset = load_image_dataset(name, self.fwd_trans, is_train)

    def __getitem__(self, idx) -> tuple:
        image = self.dataset[idx][0]
        ctxt = 0
        return image, ctxt

    def __len__(self) -> int:
        return len(self.dataset)

    def image_shape(self) -> list:
        return self.dataset[0][0].shape
