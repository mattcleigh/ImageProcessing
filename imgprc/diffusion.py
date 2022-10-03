
import math

import numpy as np

import torch as T
import torch.nn as nn
from torch.nn.functional import softmax, avg_pool2d, interpolate

from mattstools.cnns import UNet
from mattstools.network import MyNetBase
from mattstools.modules import DenseNetwork
from mattstools.torch_utils import get_loss_fn, to_np

import wandb

class CosineBetaSchedule:
    """Anneal the beta_t evolution using a cosine schedule

    Proposed in https://arxiv.org/abs/2102.09672
    """
    def __init__(self, n_timesteps: int, s: float=1e-2):
        """
        args:
            n_timesteps: The total number of timesteps used in the diffusion process
        kwargs:
            s: Smoothing parameter for the cosine annealing
        """
        t = T.linspace(0, n_timesteps, n_timesteps+1)
        f = T.cos(((t / n_timesteps) + s) / (1 + s) * T.pi * 0.5) ** 2
        self.alphabars = f / f[0]
        self.betas = 1 - (self.alpha_bars[1:] / self.alpha_bars[:-1])

    def get_betas(self, times: T.Tensor)->T.Tensor:
        """Return the beta values given specific timesteps, works with batch dim"""
        return self.betas[times]

    def get_alphabars(self, times: T.Tensor)->T.Tensor:
        """Return the alpabar values given specific timesteps, works with batch dim"""
        return self.alpha_bars[times]

    def get_recip_sqrt_alphabars(self, times: T.Tensor)->T.Tensor:
        """Return the recoprocal root alpha var values given specific timesteps"""
        return 1 / self.alpha_bars[times].sqrt()


class SinusoidalEmbedding(nn.Module):
  """Computes a positional embedding of timesteps using a sinusoid.
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

  def forward(self, x: T.Tensor)->T.Tensor:
    """Return the embedding for given positions, works with batch dimension"""
    half_dim = self.dim // 2
    emb = (- T.arange(half_dim, device=x.device) * math.log(10000) / half_dim).exp()
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
        n_timesteps: int = 1000,
        time_embedding_dim: int = 8,
        upscale_factor: int = 4,
        unet_kwargs: dict = None,
        time_embed_kwargs: dict = None,
    ) -> None:
        """
        kwargs:
            loss_name: Name of the loss function to use for noise estimation
            n_timesteps: Total number of timesteps used in the diffusion process
            time_embedding_dim: Embedding size of the timestep encoding
            upscale_factor: Amount by which to upscale the image
            unet_kwargs: Keyword arguments for the UNet network
            time_embed_kwargs: Keyword arguments for the Dense timestep embedder
        """
        super().__init__(**base_kwargs)

        ## Safe dict default kwargs
        unet_kwargs = unet_kwargs or {}

        ## Attributes
        self.n_timesteps = n_timesteps
        self.time_embedding_dim = time_embedding_dim
        self.upscale_factor = upscale_factor
        self.loss_fn = get_loss_fn(loss_name)

        ## The embedding class used to turn the timestep into a context vector
        self.time_embedder = nn.Sequential(
            SinusoidalEmbedding(self.time_embedding_dim),
            DenseNetwork(
                inpt_dim=self.time_embedding_dim,
                outp_dim=4*self.time_embedding_dim,
                **time_embed_kwargs
                ),
        )

        ## Initialise the generator model making up this network
        self.unet = UNet(
            inpt_size=self.inpt_dim[1:],
            inpt_channels=self.inpt_dim[0]*2, ## Will concatenate blurry image to inpt
            outp_channels=self.inpt_dim[0],
            ctxt_dim=4*self.time_embedding_dim,
            **unet_kwargs,
        )

        ## The scheduler for calculating the appropriate beta values
        self.beta_shedule = CosineBetaSchedule(self.n_timesteps)

        ## Initialise the optimisers
        self._setup()

    def get_losses(
        self, sample: tuple, _batch_idx: int = None, _epoch_num: int = None
    ) -> dict:
        """Function called by the trainer"""

        ## Unpack the sample tuple
        images, ctxt_tensor = sample

        ## Downscale and upscale the images to be used for conditional diffusion
        ctxt_images = interpolate(
            avg_pool2d(images, self.upscale_factor, self.upscale_factor),
            scale_factor=self.upscale_factor
        )

        ## Sample random timesteps and noise values
        times = T.randint(low=1, high=self.n_timesteps, device=self.device)
        noise = T.randn_like(ctxt_images)

        ## Retrive the alpha bar values from the beta scheduler
        alpha_bars = self.beta_shedule.get_alphabar_values(times)

        ## Apply the noise to the input images using the diffusion equation
        x_t = alpha_bars.sqrt() * images + (1-alpha_bars).sqrt() * noise

        ## Estimate the noise that was added using the conditional UNet
        estimated_noise = self.estimate_noise(x_t, ctxt_images, times)

        ## Calculate the loss on the noise values
        rec_loss = self.loss_fn(estimated_noise, images)

        ## Return the loss dictionary
        return {"total": rec_loss}

    def estimate_noise(self, x_t: T.Tensor, ctxt_image: T.Tensor, time: T.Tensor):
        """
        args:
            x_t: The current iteration of the diffusion process
            ctxt_image: The FULL size but blurry image for context
            time: The iteration point, used through positional encoding
        """
        time_encoding = self.time_embedder(time)
        return self.unet.forward(T.cat([x_t, ctxt_image], dim=1), time_encoding)

    def ddpm_denoise(self, x_t: T.Tensor, ctxt_image: T.Tensor, time: T.Tensor):
        """Apply a single denoising step for a particular point on the trajectory"""

        ## Calculate the terms from the scheduler
        betas = self.beta_shedule.get_betas(time)
        alphabars = self.beta_shedule.get_alphabars(time)
        noise = T.randn_like(x_t)

        ## Get the estimated noise using the network
        estimated_noise = self.estimate_noise(x_t, ctxt_image, time)

        ## Multiply the estimated noise values by their coefficients
        return (x_t - betas / (1-alphabars).sqrt()) / alphabars.sqrt() + betas * noise

    def ddim_denoise(self, x_t: T.Tensor, ctxt_image: T.Tensor, time: T.Tensor, time_next: T.Tensor):
        """Apply a single denoising step using the DDIM method
        Here time_next is the next iteration point we will jump straight to
        """

        ## Calculate the terms from the scheduler
        alphabars = self.beta_shedule.get_alphabars(time)
        next_alphabars = self.beta_shedule.get_alphabars(time_next)

        ## Get the estimated noise using the network
        estimated_noise = self.estimate_noise(x_t, ctxt_image, time)

        ## Return the DDIM jump step
        out = (x_t - T.sqrt(1-alphabars)*estimated_noise)
        out *= T.sqrt(next_alphabars/alphabars)
        out += T.sqrt(1-next_alphabars)*estimated_noise
        return out

    def sample(self, ctxt_image: T.Tensor, n_steps: None, keep_all: bool = False):
        """Apply the full reverse process to the image
        args:
            ctxt_image: The FULL size but blurry image for context
        kwargs:
            n_steps: The number of iterations to generate the images
            keep_all: If true then it will return all images (memory heavy!)
        """

        ## Check the input argument for the n_steps, must be less than what was trained
        all_outs = []
        batch_size = ctxt_image.shape[0]
        n_steps = n_steps or self.n_timesteps
        assert n_steps <= self.n_timesteps

        ## Sample under the prior
        x_t = T.rand_like(ctxt_image)

        ## If using the full range, then we use DDPM
        if n_steps == self.n_timesteps:
            for time in reversed(range(1, self.n_timesteps)):

                ## Expand the times to the full batch size
                time = T.tensor([time], device=self.device).repeat(batch_size)

                ## Perform the denoising step
                x_t = self.ddpm_denoise(x_t, ctxt_image, time)
            return x_t

        ## If using less timesteps then we use the DDIM sampling method
        time_steps = range(0, self.n_timesteps, self.n_timesteps//n_steps)
        time_steps = list(time_steps)
        if time_steps[-1] != self.n_timesteps:
            time_steps.append(self.n_timesteps)
        time_steps.reverse()
        for time, time_next in zip(time_steps[:-1], time_steps[1:]):
            time = T.tensor([time], device=self.device).repeat(batch_size)
            time_next = T.tensor([time_next], device=self.device).repeat(batch_size)
            x_t = self.ddim_denoise(x_t, ctxt_image, time, time_next)
        return x_t

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
        compressed_images = interpolate(
            avg_pool2d(images, self.upscale_factor, self.upscale_factor),
            scale_factor=self.upscale_factor,
        )

        ## Sample from the diffusion process
        outputs = self.sample(compressed_images, n_steps=50)

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
