"""
Collection of pytorch networks for flavour tagging
"""

# pylint: disable=unsubscriptable-object
from dotmap import DotMap

import numpy as np

import torch as T
import torch.nn as nn
from torch.nn.functional import softmax, avg_pool2d, interpolate

from mattstools.network import MyNetBase
from mattstools.torch_utils import to_np
from mattstools.cnns import DoublingConvNet, UNet

import wandb


class CNNClassifier(MyNetBase):
    """Classifier with unet options
    """

    def __init__(
        self, *, base_kwargs: dict, label_smoothing: float = 0, cnn_kwargs: dict = None
    ) -> None:
        """
        kwargs:
            label_smoothing: Amount of label smoothing to apply
            cnn_kwargs: Keyword arguments for the CNN network
        """
        super().__init__(**base_kwargs)

        ## Safe dict default kwargs
        cnn_kwargs = cnn_kwargs or {}

        ## Initialise the main model making up this network
        self.cnn = DoublingConvNet(
            inpt_size=self.inpt_dim[1:],
            inpt_channels=self.inpt_dim[0],
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_dim,
            **cnn_kwargs,
        )

        ## Declare the names of the metrics to keep track of
        self.loss_names = ["total", "accuracy"]

        ## The loss function for classification
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._setup()

    def get_losses(
        self, sample: tuple, _batch_idx: int = None, _epoch_num: int = None
    ) -> dict:
        """Function called by the trainer"""

        ## Unpack the sample tuple
        images, ctxt, labels = sample

        ## Get the outputs
        outputs = self.cnn(images, ctxt)

        ## Calculate the image classification loss
        cls_loss = self.loss_fn(outputs, labels)

        ## Calculate the accuracy
        accuracy = (T.argmax(outputs, dim=-1) == labels).float().mean()

        ## Return the loss dictionary
        return {"total": cls_loss, "accuracy": accuracy}

    def forward(self, images: T.Tensor, ctxt: T.Tensor = None):
        return self.cnn(images, ctxt)

    def get_scores(self, images: T.Tensor, ctxt: T.Tensor = None):
        return softmax(self.cnn(images, ctxt), dim=-1)

    def visualise(self, loader, path, flag, epochs):
        """Visualise the predictions using weights and biases"""

        ## Only works if wandb is currently active
        if wandb.run is None:
            return

        ## Create the wandb table
        columns = ["id", "image", "label", "pred", "scores"]
        test_table = wandb.Table(columns=columns)

        ## Just take the first batch
        classes = loader.dataset.dataset.classes
        images, ctxt, labels = next(iter(loader))
        scores = self.get_scores(images.to(self.device), ctxt.to(self.device))

        ## Convert to numpy
        images = to_np(images)
        labels = to_np(labels)
        scores = to_np(scores)
        preds = np.argmax(scores, axis=-1)

        ## Add all data to the table
        for idx, (i, l, p, s) in enumerate(zip(images, labels, preds, scores)):
            img_id = str(idx) + "_" + str(epochs)
            test_table.add_data(
                img_id,
                wandb.Image(np.transpose(i, (1, 2, 0))),
                classes[l],
                classes[p],
                s,
            )

        ## Sync the table with wand
        wandb.run.log({"test_predictions": test_table}, step=epochs - 1, commit=False)


class UNetSuperResolution(MyNetBase):
    """A image to image model for doubling the resolution of an image
    """

    def __init__(self, *, base_kwargs: dict, unet_kwargs: dict = None) -> None:
        """
        kwargs:
            unet_kwargs: Keyword arguments for the CNN network
        """
        super().__init__(**base_kwargs)

        ## Safe dict default kwargs
        unet_kwargs = unet_kwargs or {}

        ## Initialise the main model making up this network
        self.unet = UNet(
            inpt_size=self.inpt_dim[1:],
            inpt_channels=self.inpt_dim[0],
            outp_channels=self.inpt_dim[0],
            ctxt_dim=self.ctxt_dim,
            **unet_kwargs,
        )

        ## Declare the names of the metrics to keep track of
        self.loss_names = ["total"]

        ## The loss function for classification
        self.loss_fn = nn.MSELoss()
        self._setup()

    def get_losses(
        self, sample: tuple, _batch_idx: int = None, _epoch_num: int = None
    ) -> dict:
        """Function called by the trainer"""

        ## Unpack the sample tuple
        images, ctxt = sample

        ## Calculate the image reconstruction loss
        rec_loss = self.loss_fn(self.forward(images, ctxt), images)

        ## Return the loss dictionary
        return {"total": rec_loss}

    def forward(self, images: T.Tensor, ctxt: T.Tensor = None):
        in_images = avg_pool2d(images, 4, 4)
        in_images = interpolate(in_images, scale_factor=4)
        rec_images = self.unet(in_images, ctxt)
        return rec_images

    def visualise(self, loader, path, flag, epochs):
        """Visualise the predictions using weights and biases"""

        ## Only works if wandb is currently active
        if wandb.run is None:
            return

        ## Create the wandb table
        columns = ["idx", "input", "output", "truth"]
        test_table = wandb.Table(columns=columns)

        ## Just take the first batch
        images, ctxt = next(iter(loader))

        ## Convert to numpy
        inputs = to_np(avg_pool2d(images, 4, 4))
        outputs = to_np(self.forward(images.to(self.device), ctxt.to(self.device)))
        truth = to_np(images)

        ## Add all data to the table
        for idx, (i, o, t) in enumerate(zip(inputs, outputs, truth)):
            img_id = str(idx) + "_" + str(epochs)
            test_table.add_data(
                img_id,
                wandb.Image(np.transpose(i, (1, 2, 0))),
                wandb.Image(np.transpose(o, (1, 2, 0))),
                wandb.Image(np.transpose(t, (1, 2, 0))),
            )

        ## Sync the table with wand
        wandb.run.log({"test_predictions": test_table}, step=epochs - 1, commit=False)
