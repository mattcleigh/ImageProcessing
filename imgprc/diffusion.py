import math
from typing import Tuple

from tqdm import tqdm
import numpy as np

import torch as T
import torch.nn as nn
from torch.nn.functional import softmax, avg_pool2d, interpolate

from mattstools.cnns import UNet
from mattstools.network import MyNetBase
from mattstools.modules import DenseNetwork
from mattstools.torch_utils import get_loss_fn, to_np

import wandb


def diffusion_shedule(
    diff_time: T.Tensor, max_sr: float = 1, min_sr: float = 1e-8
) -> Tuple[T.Tensor, T.Tensor]:
    """
    We will use continuous diffusion times between 0 and 1 which make switching between
    different numbers of diffusion steps between training and testing much easier

    Returns only the values needed for the jump forward diffusion step and the reverse
    DDIM step.
    These are sqrt(alpha_bar) and sqrt(1-alphabar) which are called the signal_rate
    and noise_rate respectively.

    The jump forward diffusion process is simply a weighted sum of:
        input * signal_rate + eps * noise_rate

    Uses a cosine annealing schedule as proposed in
    Proposed in https://arxiv.org/abs/2102.09672

    args:
        diff_time: The time used to sample the diffusion scheduler
            Output will match the shape
            Must be between 0 and 1
    kwargs:
        max_sr: The initial rate at the first step
        min_sr: How much signal is preserved at end of diffusion
            (can't be zero due to log)
    """

    ## Use cosine annealing, which requires switching from times -> angles
    start_angle = math.acos(max_sr)
    end_angle = math.acos(min_sr)
    diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
    signal_rates = T.cos(diffusion_angles)
    noise_rates = T.sin(diffusion_angles)
    return signal_rates, noise_rates


class SinusoidalEmbedding(nn.Module):
    """Computes a positional embedding of time values using a sinusoid.
  This is the same equation used to for positional encodings for transformers

  The embedded vector is [ sin(w_0*t), ... sin(w_d/2*t), cos(w_0*t) ... cos(w_d/2*t)]
  Where:
  - d is the embedded dimension
  - w_k is the specific frequency = 1/(10000)^(2k/d)
  """

    def __init__(self, dim: int):
        """
    args:
        dim: The embedding dimension size, must be even
    """
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Return the embedding for given positions, works with batch dimension"""
        half_dim = self.dim // 2
        emb = (-T.arange(half_dim, device=x.device) * math.log(10000) / half_dim).exp()
        emb = x.view(-1, 1) * emb.view(-1, half_dim)
        emb = T.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionSuperResolution(MyNetBase):
    """A image to image model for increasing the resolution of an image using the
    diffusion process.
    """

    def __init__(
        self,
        *,
        base_kwargs: dict,
        loss_name: str = "mse",
        time_embedding_dim: int = 8,
        upscale_factor: int = 4,
        unet_kwargs: dict = None,
        time_embed_kwargs: dict = None,
    ) -> None:
        """
        kwargs:
            loss_name: Name of the loss function to use for noise estimation
            time_embedding_dim: Embedding size of the diffusion time encoding
            upscale_factor: Amount by which to upscale the image
            unet_kwargs: Keyword arguments for the UNet network
            time_embed_kwargs: Keyword arguments for the Dense time embedder
        """
        super().__init__(**base_kwargs)

        ## Safe dict default kwargs
        unet_kwargs = unet_kwargs or {}

        ## Attributes
        self.time_embedding_dim = time_embedding_dim
        self.upscale_factor = upscale_factor
        self.loss_fn = get_loss_fn(loss_name)

        ## The embedding class used to turn the timestep into a context vector
        self.time_embedder = nn.Sequential(
            SinusoidalEmbedding(self.time_embedding_dim),
            DenseNetwork(
                inpt_dim=self.time_embedding_dim,
                outp_dim=4 * self.time_embedding_dim,
                **time_embed_kwargs,
            ),
        )

        ## Initialise the generator model making up this network
        self.unet = UNet(
            inpt_size=self.inpt_dim[1:],
            inpt_channels=self.inpt_dim[0]
            * 2,  ## Will concatenate blurry image to inpt
            outp_channels=self.inpt_dim[0],
            ctxt_dim=4 * self.time_embedding_dim,
            **unet_kwargs,
        )

        ## Initialise the optimisers
        self._setup()

    def get_losses(
        self, sample: tuple, _batch_idx: int = None, _epoch_num: int = None
    ) -> dict:
        """Function called by the trainer"""

        ## Unpack the sample tuple
        images, _ = sample

        ## Downscale and upscale the images to be used for conditional diffusion
        ctxt_images = interpolate(
            avg_pool2d(images, self.upscale_factor, self.upscale_factor),
            scale_factor=self.upscale_factor,
        )

        ## Sample random time values and get the noise rates from the scheduler
        diff_time = T.rand(size=(len(images), 1), device=self.device)
        signal_rates, noise_rates = diffusion_shedule(diff_time.view(-1, 1, 1, 1))

        ## Sample random noise to perturb the images
        noise = T.randn_like(ctxt_images)

        ## Apply the noise to the input images using the diffusion equation
        x_t = signal_rates * images + noise_rates * noise

        ## Predict the noise that was added using the conditional UNet
        pred_noise = self.pred_noise(x_t, ctxt_images, diff_time)

        ## Calculate the loss on the noise values
        rec_loss = self.loss_fn(pred_noise, noise).mean()

        ## Return the loss dictionary
        return {"total": rec_loss}

    def pred_noise(
        self, x_t: T.Tensor, ctxt_image: T.Tensor, time: T.Tensor
    ) -> T.Tensor:
        """Predict the noise applied to an image at a specific time in the diffusion
        args:
            x_t: The current iteration of the diffusion process
            ctxt_image: The FULL size context image
            time: The iteration time between 0 and 1
        """
        return self.unet(T.cat([x_t, ctxt_image], dim=1), ctxt=self.time_embedder(time))

    def reverse_diffusion(
        self, ctxt_image: T.Tensor, n_steps: 50, keep_all: bool = False
    ) -> Tuple[T.Tensor, list]:
        """Apply the full reverse process to the image
        args:
            ctxt_image: The FULL size but blurry image for context
        kwargs:
            n_steps: The number of iterations to generate the images
            keep_all: If true then it will return all images (memory heavy!)
        """

        ## Check the input argument for the n_steps, must be less than what was trained
        all_x = []
        batch_size = ctxt_image.shape[0]
        step_size = 1 / n_steps

        ## Start the iteration by sampling under the prior
        x_t = T.rand_like(ctxt_image)
        for step in tqdm(range(n_steps), "generating"):

            ## Keep track of the diffusion evolution
            if keep_all:
                all_x.append(x_t)

            ## Get the diffusion time working back from 1 to 0
            diff_time = T.ones((batch_size,), device=self.device) - step * step_size

            ## Predict x_0 using the DIMM method
            sr, nr = diffusion_shedule(diff_time.view(-1, 1, 1, 1))
            pred_noise = self.pred_noise(x_t, ctxt_image, diff_time)
            x_0 = (x_t - nr * pred_noise) / sr

            ## The final step wont be used
            if step == (n_steps-1):
                break

            ## Remix the predicted components using next signal and noise rates
            next_diff_time = diff_time - step_size
            next_sr, next_nr = diffusion_shedule(next_diff_time.view(-1, 1, 1, 1))
            x_t = next_sr * x_0 + next_nr * pred_noise

        return x_0, all_x

    def visualise(self, loader, path, flag, epochs):
        """Visualise the predictions using weights and biases"""

        ## Only works if wandb is currently active
        if wandb.run is None:
            return

        ## Create the wandb table
        columns = ["idx", "input", "output", "truth"]
        test_table = wandb.Table(columns=columns)

        ## Just take the first batch
        images, _ = next(iter(loader))

        ## Get the low quality input images for the network
        compressed_images = interpolate(
            avg_pool2d(images, self.upscale_factor, self.upscale_factor),
            scale_factor=self.upscale_factor,
        )

        ## Sample from the diffusion process
        outputs, _ = self.reverse_diffusion(compressed_images.to(self.device), n_steps=100)

        ## Clip between -1 and 1 for showing images using floats
        outputs = T.clamp(outputs, -1, 1)

        ## Convert to numpy
        inputs = to_np(compressed_images)
        outputs = to_np(outputs)
        truth = to_np(images)

        ## Add all data to the table
        for idx, (i, o, t) in enumerate(zip(inputs, outputs, truth)):
            test_table.add_data(
                idx,
                wandb.Image(np.transpose(i, (1, 2, 0))),
                wandb.Image(np.transpose(o, (1, 2, 0))),
                wandb.Image(np.transpose(t, (1, 2, 0))),
            )

        ## Sync the table with wand
        wandb.run.log({"test_predictions": test_table}, commit=False)
