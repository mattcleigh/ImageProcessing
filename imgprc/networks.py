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
from mattstools.torch_utils import get_optim, get_sched, GradsOff

import wandb

from imgprc.modules import PatchDiscriminator
from imgprc.loss import GANLoss


class CNNClassifier(MyNetBase):
    """Classifier with unet options"""

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
    """A image to image model for doubling the resolution of an image"""

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

        ## Get the low quality input images for the network
        in_images = avg_pool2d(images, 4, 4)

        ## Calculate the image reconstruction loss
        rec_loss = self.loss_fn(self.forward(in_images, ctxt), images)

        ## Return the loss dictionary
        return {"total": rec_loss}

    def forward(self, images: T.Tensor, ctxt: T.Tensor = None):
        """Takes in a 32x32 image and upscales it to 128x128"""
        ## Use nearest neibour upscaling for inputs, then pass through network
        images = interpolate(images, scale_factor=4)
        return self.unet(images, ctxt)

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

        ## Get the low quality input images for the network
        in_images = avg_pool2d(images, 4, 4)

        ## Get the network outputs
        outputs = self.forward(in_images.to(self.device), ctxt.to(self.device))

        ## Convert to numpy
        inputs = to_np(in_images)
        outputs = to_np(outputs)
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


class UNetSRGAN(MyNetBase):
    """A image to image model for quadupling the resolution of an image, with an
    additional adversay.

    GANs in mattstools need to handle their own optimisers using
    train and validation steps.
    """

    def __init__(
        self,
        *,
        base_kwargs: dict,
        steps_per_epoch: int = 0,
        gan_mode: str = "vanilla",
        grad_clip: int = 10,
        unet_kwargs: dict = None,
        disc_kwargs: dict = None,
        optim_dict: dict = None,
        sched_dict: dict = None,
    ) -> None:
        """
        args:
            steps_per_epoch: Needed as the model must configure its own scheduler
            grad_clip: Needed as the model must perform its own loss
        kwargs:
            gan_mode: loss type to use for the GAN
            unet_kwargs: Keyword arguments for the UNet network
            disc_kwargs: Keyword arguments for the CNN discriminator network
            optim_dict: Keyword arguments for the optimisers
            sched_dict: Keyword arguments for the schedulers
        """
        super().__init__(**base_kwargs)

        ## Safe dict default kwargs
        unet_kwargs = unet_kwargs or {}
        disc_kwargs = disc_kwargs or {}
        optim_dict = optim_dict or {}
        sched_dict = sched_dict or {}

        ## Initialise the generator model making up this network
        self.unet = UNet(
            inpt_size=self.inpt_dim[1:],
            inpt_channels=self.inpt_dim[0],
            outp_channels=self.inpt_dim[0],
            ctxt_dim=self.ctxt_dim,
            **unet_kwargs,
        )

        ## Initialise the discriminator model making up this network
        self.disc = PatchDiscriminator(
            inpt_size=self.inpt_dim[1:],
            inpt_channels=self.inpt_dim[0],
            ctxt_dim=self.ctxt_dim,
            **disc_kwargs,
        )

        ## Initialise the optimisers
        self.grad_clip = grad_clip
        self.g_opt = get_optim(optim_dict, self.unet.parameters())
        self.d_opt = get_optim(optim_dict, self.disc.parameters())

        ## Initialise the learning rate schedulers
        self.g_sched = get_sched(sched_dict, self.g_opt, steps_per_epoch)
        self.d_sched = get_sched(sched_dict, self.d_opt, steps_per_epoch)

        ## Declare the names of the metrics to keep track of
        self.loss_names = ["total", "reconstruction", "generator", "discriminator"]

        ## The loss function for reconstruction
        self.rec_loss_fn = nn.L1Loss()
        self.gan_loss_fn = GANLoss(gan_mode)
        self._setup()

    def _step(
        self,
        is_train: bool,
        sample: tuple,
        _batch_idx: int = None,
        _epoch_num: int = None,
    ):
        """Function called by trainer"""

        ## Unpack the sample tuple
        images, ctxt = sample

        ## Get the low quality input images for the network
        in_images = avg_pool2d(images, 4, 4)
        in_images = interpolate(in_images, scale_factor=4)

        #################
        ## G optim step
        #################

        ## Get the upscaled images, pass through dist and calc loss
        out_images = self.unet(in_images, ctxt)
        with GradsOff(self.disc):
            disc_outs = self.disc(out_images, ctxt)
        rec_loss = self.rec_loss_fn(out_images, in_images)
        gan_loss = self.gan_loss_fn(disc_outs, True)  ## Gen uses wrong labels
        gen_loss = rec_loss + gan_loss

        ## Perform the step for the generator
        if is_train:
            self.g_opt.zero_grad(set_to_none=True)
            gen_loss.backward()
            nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
            self.g_opt.step()
            self.g_sched.step()

        #################
        ## D optim step
        #################

        ## Calculate the loss for the discriminator
        fake_loss = self.gan_loss_fn(self.disc(out_images.detach(), ctxt), False)
        real_loss = self.gan_loss_fn(self.disc(images, ctxt), True)
        disc_loss = (fake_loss + real_loss) / 2

        ## Perform the step for the discriminator
        if is_train:
            self.d_opt.zero_grad(set_to_none=True)
            disc_loss.backward()
            nn.utils.clip_grad_norm_(self.disc.parameters(), self.grad_clip)
            self.d_opt.step()
            self.d_sched.step()

        ## Return the loss names for plotting
        return {
            "total": gen_loss,
            "reconstruction": rec_loss,
            "generator": gan_loss,
            "discriminator": disc_loss,
        }

    def train_step(self, *args, **kwargs):
        """Called by the trainer for updating the optimisers"""
        return self._step(True, *args, **kwargs)

    def valid_step(self, *args, **kwargs):
        """Called by the trainer for the validation step"""
        return self._step(False, *args, **kwargs)

    def forward(self, images: T.Tensor, ctxt: T.Tensor = None):
        """Takes in a 32x32 image and upscales it to 128x128"""
        ## Use nearest neibour upscaling for inputs, then pass through network
        images = interpolate(images, scale_factor=4)
        return self.unet(images, ctxt)

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

        ## Get the low quality 32x32 input images for the network
        in_images = avg_pool2d(images, 4, 4)

        ## Get the network outputs 128x128
        outputs = self.forward(in_images.to(self.device), ctxt.to(self.device))

        ## Convert to numpy
        inputs = to_np(in_images)
        outputs = to_np(outputs)
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
