from copy import deepcopy
from functools import partial
from typing import Mapping

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize


class ImageUpscaleDataset(Dataset):
    """A class which returns the image and a downscaled copy for upscaling."""

    def __init__(self, dataset: partial, factor: int = 2) -> None:
        super().__init__()
        self.dataset = dataset()
        self.factor = factor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        image, ctxt = self.dataset[idx]
        orig_size = np.array(image.shape[1:])
        low_size = orig_size // self.factor
        low_res = resize(image, size=low_size, antialias=True)
        low_res = resize(low_res, size=orig_size, antialias=True)
        return image, ctxt, low_res


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        test_set: partial,
        train_set: partial,
        loader_config: Mapping,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load the datasets
        self.test_set = test_set()  # For now test=valid

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""

        if stage in ["fit"]:
            self.train_set = self.hparams.train_set()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_config)

    def test_dataloader(self) -> DataLoader:
        test_config = deepcopy(self.hparams.loader_config)
        test_config["drop_last"] = False
        test_config["shuffle"] = False
        return DataLoader(self.test_set, **test_config)

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_data_sample(self) -> tuple:
        """Get a single data sample to help initialise the neural network."""
        return self.test_set[0]
