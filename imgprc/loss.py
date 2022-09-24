"""
Custom loss functions and classes to use in the setup
"""

import torch as T
import torch.nn as nn


class GANLoss(nn.Module):
    """Defines different variant on the GAN loss

    Also includes the target generation of ones and zeros based on classes
    Discriminator outputs must be the raw logits! Don't use a sigmoid layer!
    """

    def __init__(
        self,
        gan_mode: str = "vanilla",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ):
        """
        args:
            gan_mode: Type of GAN loss, vanilla (nst), lsgan, and wgan.
            real_label: The class label for a real image
            fake_label: The class label for a real image
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()

        ## Class attributes (buffers move with device)
        self.gan_mode = gan_mode
        self.register_buffer("real_label", T.tensor(real_label))
        self.register_buffer("fake_label", T.tensor(fake_label))

        ## Determin the type of loss function needed
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode in "wgan":  ## Wasertein gans have no loss, just push outputs
            self.loss = None
        else:
            raise NotImplementedError(f"Unrecognised gan_mode: {gan_mode}")

    def get_target_tensor(self, prediction: T.Tensor, target_is_real: bool) -> T.Tensor:
        """Create label tensors with the same size as the discriminators predictions
        args:
            prediction: The output of the discriminator
            target_is_real: If these images should be considered as real or fake
        returns:

        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction: T.Tensor, target_is_real: bool) -> T.Tensor:
        """Calculate loss given discriminator's output and grount truth labels.
        args:
            prediction: The output of the discriminator
            target_is_real: If these images should be considered as real or fake
        returns:
            the calculated loss averaged over the batch
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgan":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
