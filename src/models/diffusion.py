import copy
from functools import partial
from typing import Mapping, Tuple

import numpy as np
import pytorch_lightning as pl
import torch as T
import wandb

from mattstools.mattstools.k_diffusion import get_timesteps, heun_sampler
from mattstools.mattstools.modules import CosineEncoding, IterativeNormLayer
from mattstools.mattstools.torch_utils import get_loss_fn, get_sched
from mattstools.mattstools.cnns import UNet
from torchvision.utils import make_grid

class ImageDiffusionGenerator(pl.LightningModule):
    """A generative model which uses the diffusion process on an image input."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        cosine_config: Mapping,
        normaliser_config: Mapping,
        unet_config: Mapping,
        optimizer: partial,
        sched_config: Mapping,
        use_ctxt: bool = True,
        loss_name: str = "mse",
        min_time: float = 0,
        max_time: float = 80.0,
        ema_sync: float = 0.999,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sampler_name: str = "heun",
        sampler_steps: int = 50,
        sampler_min_time: int = 1e-3,
        sampler_curvature: float = 7.0,
    ) -> None:
        """
        Parameters:
        -----------
        data_sample : tuple
            A tuple containing the input and context data samples.
        cosine_config : Mapping
            A dictionary containing the configuration settings for the cosine encoding.
        normaliser_config : Mapping
            A dictionary containing the configuration settings for the iterative
            normalization layer.
        unet_config : Mapping
            A dictionary containing the configuration settings for the UNet.
        optimizer : partial
            The optimizer function used to optimize the neural network.
        sched_config : Mapping
            A dictionary containing the configuration settings for the scheduler.
        use_ctxt : Bool
            If the config from the image is used in the diffusion process
            Default is False
        loss_name : str, optional
            The name of the loss function used to train the neural network.
            Default is "mse".
        min_time : float, optional
            The minimum time value used in the diffusion training.
            Default is 0.
        max_time : float, optional
            The maximum time value used in the diffusion training.
            Default is 80.0.
        ema_sync : float, optional
            The exponential moving average sync value. Default is 0.999.
        p_mean : float, optional
            The mean value used for the sigma distribution during training.
            Default is -1.2.
        p_std : float, optional
            The standard deviation value used for the sigma distribution during
            training.
            Default is 1.2.
        sampler_name : str, optional
            The name of the sampler used in the validation/testing loop.
            Default is "heun".
        sampler_steps : int, optional
            The number of steps used for the sampler in the validation/testing loop.
            Default is 50.
        sampler_min_time : int, optional
            The minimum time value used for the sampler in the validation/testing loop.
            Default is 1e-3.
        sampler_curvature : float, optional
            The curvature value used for the sampler in the validation/testing loop.
            Default is 7.0.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample to get the important dimensions
        self.inpt_dim = data_sample[0].shape
        self.ctxt_dim = data_sample[1].shape if use_ctxt else 0

        # Class attributes
        self.loss_fn = get_loss_fn(loss_name)
        self.ema_sync = ema_sync
        self.min_time = min_time
        self.max_time = max_time
        self.p_mean = p_mean
        self.p_std = p_std
        self.use_ctxt = use_ctxt

        # The encoder and scheduler needed for diffusion
        self.time_encoder = CosineEncoding(
            min_value=0, max_value=max_time, **cosine_config
        )

        # The layer which normalises the input point cloud data
        self.normaliser = IterativeNormLayer(
            self.inpt_dim, extra_dims=(1, 2), **normaliser_config
        )
        if self.ctxt_dim:
            self.ctxt_normaliser = IterativeNormLayer(
                self.ctxt_dim, **normaliser_config
            )

        # The base UNet
        self.net = UNet(
            inpt_size=self.inpt_dim[1:],
            inpt_channels=self.inpt_dim[0],
            outp_channels=self.inpt_dim[0],
            ctxt_dim=self.ctxt_dim + self.time_encoder.outp_dim,
            **unet_config,
        )

        # A copy of the network which will sync with an exponential moving average
        self.ema_net = copy.deepcopy(self.net)

        # Sampler to run in the validation/testing loop
        self.sampler_curvature = sampler_curvature
        self.sampler_name = sampler_name
        self.sampler_min_time = sampler_min_time
        self.sampler_steps = sampler_steps

        # Initial noise for running the visualisation at the end of the epoch
        self.initial_noise = T.randn((5, *self.inpt_dim)) * self.max_time

    def get_c_values(self, diffusion_times: T.Tensor) -> tuple:
        """Calculate the c values needed for the I/O"""

        # Ee use cos encoding so we dont need c_noise
        c_in = 1 / (1 + diffusion_times**2).sqrt()
        c_out = diffusion_times / (1 + diffusion_times**2).sqrt()
        c_skip = 1 / (1 + diffusion_times**2)

        return c_in, c_out, c_skip

    def denoise(
        self,
        noisy_data: T.Tensor,
        diffusion_times: T.Tensor,
        ctxt: T.Tensor | None = None,
        mask: T.BoolTensor | None = None,  # Keep this for generalisation for pc gen
    ) -> T.Tensor:
        """Return the denoised data from a given timestep."""

        # Get the c values for the data scaling
        c_in, c_out, c_skip = self.get_c_values(diffusion_times.view(-1, 1, 1, 1))

        # Scale the inputs and pass through the network
        outputs = self.forward(c_in * noisy_data, diffusion_times, ctxt)

        # Get the denoised output by passing the scaled input through the network
        return c_skip * noisy_data + c_out * outputs

    def forward(
        self,
        noisy_data: T.Tensor,
        diffusion_times: T.Tensor,
        ctxt: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass through the model, corresponds to F_theta in the Karras
        paper."""

        # Use the appropriate network for training or validation
        if self.training:
            network = self.net
        else:
            network = self.ema_net

        # Encode the times and combine with existing context info
        context = self.time_encoder(diffusion_times)
        if self.ctxt_dim:
            context = T.cat([context, ctxt], dim=-1)

        # Use the selected network to esitmate the noise present in the data
        return network(noisy_data, ctxt=context)

    def _shared_step(self, sample: tuple) -> Tuple[T.Tensor, T.Tensor]:
        """Shared step used in both training and validaiton."""

        # Unpack the sample tuple
        data, ctxt = sample

        # Pass through the normalisers
        data = self.normaliser(data)
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)

        # Sample times using the Karras method using a log normal distribution
        diffusion_times = T.zeros(size=(len(data), 1), device=self.device)
        diffusion_times.add_(self.p_mean + self.p_std * T.randn_like(diffusion_times))
        diffusion_times.exp_().clamp_(self.min_time, self.max_time)

        # Get the c values for the data scaling
        c_in, c_out, c_skip = self.get_c_values(diffusion_times.view(-1, 1, 1, 1))

        # Sample from N(0, t)
        noises = T.randn_like(data) * diffusion_times.view(-1, 1, 1, 1)

        # Make the noisy samples by mixing with the real data
        noisy_data = data + noises

        # Pass through the network
        output = self.forward(c_in * noisy_data, diffusion_times, ctxt)

        # Calculate the effective training target
        target = (data - c_skip * noisy_data) / c_out

        # Return the denoising loss
        return self.loss_fn(output, target).mean()

    def training_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        loss = self._shared_step(sample)
        self.log("train/total_loss", loss)
        self._sync_ema_network()
        return loss

    def validation_step(self, sample: tuple, batch_idx: int) -> None:
        loss = self._shared_step(sample)
        self.log("valid/total_loss", loss)

        # Run the generation for the first batch
        if batch_idx == 0 and wandb.run is not None:
            gen_images = self.full_generation(
                initial_noise=self.initial_noise.to(self.device), ctxt=sample[1]
            )
            wandb.log({"images": wandb.Image(make_grid(gen_images))}, commit=False)

    def _sync_ema_network(self) -> None:
        """Updates the Exponential Moving Average Network."""
        with T.no_grad():
            for params, ema_params in zip(
                self.net.parameters(), self.ema_net.parameters()
            ):
                ema_params.data.copy_(
                    self.ema_sync * ema_params.data
                    + (1.0 - self.ema_sync) * params.data
                )

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")

    @T.no_grad()
    def full_generation(
        self,
        initial_noise: T.Tensor,
        ctxt: T.Tensor | None = None,
    ) -> T.Tensor:
        """Fully generate a batch of data from noise, given context information"""

        # Normalise the context
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            assert len(ctxt) == len(initial_noise)

        # Generate the timesteps/sigma values for the sampler
        time_steps = get_timesteps(
            self.max_time, self.min_time, self.sampler_steps, self.sampler_curvature
        )

        # Run the deterministic sampler
        outputs, _ = heun_sampler(
            model=self,
            initial_noise=initial_noise,
            time_steps=time_steps,
            keep_all=False,
            ctxt=ctxt,
        )

        # Return the output
        return self.normaliser.reverse(outputs)

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.sched_config.mattstools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.sched_config.lightning},
        }
