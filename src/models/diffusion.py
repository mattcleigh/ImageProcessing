import copy
from functools import partial
from typing import Callable, Mapping, Tuple

import pytorch_lightning as pl
import torch as T
import wandb
from torchvision.utils import make_grid

from mattstools.mattstools.cnns import UNet
from mattstools.mattstools.k_diffusion import append_dims
from mattstools.mattstools.modules import CosineEncoding, IterativeNormLayer
from mattstools.mattstools.torch_utils import ema_param_sync, get_loss_fn, get_sched

# from torchmetrics.image.fid import FrechetInceptionDistance


class ImageDiffusionGenerator(pl.LightningModule):
    """A generative model which uses the diffusion process on an image
    input."""

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
        use_ctxt_img: bool = False,
        loss_name: str = "mse",
        min_sigma: float = 0,
        max_sigma: float = 80.0,
        ema_sync: float = 0.999,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sampler_function: Callable | None = None,
        sigma_function: Callable | None = None,
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
        use_ctxt_img: Bool
            If the sample context image is used for generation. Default is False
        loss_name : str, optional
            The name of the loss function used to train the neural network.
            Default is "mse".
        min_sigma : float, optional
            The minimum sigma value used in the diffusion training.
            Default is 0.
        max_sigma : float, optional
            The maximum sigma value used in the diffusion training.
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
        sampler_function : callable
            Callable sampler function to use during the generation steps
        sigma_function : callable
            Callable function for calculating the sigma steps for generation
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample to get the important dimensions
        self.inpt_dim = data_sample[0].shape
        self.ctxt_dim = data_sample[1].shape if use_ctxt else 0
        self.ctxt_img_dim = data_sample[2].shape if use_ctxt_img else [0]

        # Class attributes
        self.loss_fn = get_loss_fn(loss_name)
        self.ema_sync = ema_sync
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.p_mean = p_mean
        self.p_std = p_std
        self.use_ctxt = use_ctxt
        self.use_ctxt_img = use_ctxt_img

        # The encoder and scheduler needed for diffusion
        self.sigma_encoder = CosineEncoding(
            min_value=0, max_value=max_sigma, **cosine_config
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
            inpt_channels=self.inpt_dim[0] + self.ctxt_img_dim[0],
            outp_channels=self.inpt_dim[0],
            ctxt_dim=self.ctxt_dim + self.sigma_encoder.outp_dim,
            **unet_config,
        )

        # A copy of the network which will sync with an exponential moving average
        if ema_sync:
            self.ema_net = copy.deepcopy(self.net)
            self.ema_net.requires_grad_(False)

        # Sampler to run in the validation/testing loop
        self.sampler_function = sampler_function
        self.sigma_function = sigma_function

        # Initial noise for running the visualisation at the end of the epoch
        self.n_visualise = 5
        self.initial_noise = (
            T.randn((self.n_visualise, *self.inpt_dim)) * self.max_sigma
        )

        # For calculating the FID scores
        # self.fid = FrechetInceptionDistance(normalize=True)

    def get_c_values(self, sigmas: T.Tensor) -> tuple:
        """Calculate the c values needed for the I/O."""

        # We use cos encoding so we dont need c_noise
        c_in = 1 / (1 + sigmas**2).sqrt()
        c_out = sigmas / (1 + sigmas**2).sqrt()
        c_skip = 1 / (1 + sigmas**2)

        return c_in, c_out, c_skip

    def forward(
        self,
        noisy_data: T.Tensor,
        sigmas: T.Tensor,
        ctxt: T.Tensor | None = None,
        ctxt_img: T.Tensor | None = None,
    ) -> T.Tensor:
        """Return the denoised data from a given sigma value."""

        # Get the c values for the data scaling
        c_in, c_out, c_skip = self.get_c_values(append_dims(sigmas, noisy_data.dim()))

        # Scale the inputs and pass through the network
        outputs = self.get_outputs(c_in * noisy_data, sigmas, ctxt, ctxt_img)

        # Get the denoised output by passing the scaled input through the network
        return c_skip * noisy_data + c_out * outputs

    def get_outputs(
        self,
        noisy_data: T.Tensor,
        sigmas: T.Tensor,
        ctxt: T.Tensor | None = None,
        ctxt_img: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass through the model, corresponds to F_theta in the Karras
        paper."""

        # Use the appropriate network for training or validation
        if self.training or not self.ema_sync:
            network = self.net
        else:
            network = self.ema_net

        # Encode the sigmas and combine with existing context info
        context = self.sigma_encoder(sigmas)
        if self.ctxt_dim:
            context = T.cat([context, ctxt], dim=-1)

        # Concat the context image to the noise along the channel dimension
        if self.use_ctxt_img:
            noisy_data = T.cat([noisy_data, ctxt_img], dim=1)

        # Use the selected network to esitmate the noise present in the data
        return network(noisy_data, ctxt=context)

    def _shared_step(self, sample: tuple) -> Tuple[T.Tensor, T.Tensor]:
        """Shared step used in both training and validaiton."""

        # Unpack the sample tuple
        data = sample[0]
        ctxt = sample[1]
        ctxt_img = sample[2] if len(sample) > 2 else None

        # Pass through the normalisers
        data = self.normaliser(data)
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)

        # Sample sigmas using the Karras method of a log normal distribution
        sigmas = T.zeros(size=(data.shape[0], 1), device=self.device)
        sigmas.add_(self.p_mean + self.p_std * T.randn_like(sigmas))
        sigmas.exp_().clamp_(self.min_sigma, self.max_sigma)

        # Get the c values for the data scaling
        sigmas_with_dim = append_dims(sigmas, data.dim())
        c_in, c_out, c_skip = self.get_c_values(sigmas_with_dim)

        # Sample from N(0, sigma**2)
        noises = T.randn_like(data) * sigmas_with_dim

        # Make the noisy samples by mixing with the real data
        noisy_data = data + noises

        # Pass through the just the base network (manually scale with c values)
        output = self.get_outputs(c_in * noisy_data, sigmas, ctxt, ctxt_img)

        # Calculate the effective training target
        target = (data - c_skip * noisy_data) / c_out

        # Return the denoising loss
        return self.loss_fn(output, target).mean()

    def training_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        loss = self._shared_step(sample)
        self.log("train/total_loss", loss)
        ema_param_sync(self.net, self.ema_net, self.ema_sync)
        return loss

    def validation_step(self, sample: tuple, batch_idx: int) -> None:
        loss = self._shared_step(sample)
        self.log("valid/total_loss", loss)

        # Only if there is the logger
        if wandb.run is not None and batch_idx == 0:

            # # Unpack the sample tuple
            # data = sample[0]
            # ctxt = sample[1]
            # ctxt_img = sample[2] if len(sample) > 2 else None

            # # Fully generate a batch
            # gen_images = self.full_generation(
            #     initial_noise=T.randn_like(data),
            #     ctxt=ctxt,
            #     ctxt_img=ctxt_img,
            # ).clamp(0, 1)

            # # Calculate the FID scores
            # self.fid.update(gen_images, real=False)
            # self.fid.update(data, real=True)

            # Generate only the first few samples
            ctxt = sample[1][: self.n_visualise]
            ctxt_img = sample[2][: self.n_visualise] if len(sample) > 2 else None
            gen_images = self.full_generation(
                initial_noise=self.initial_noise.to(self.device),
                ctxt=ctxt,
                ctxt_img=ctxt_img,
            )
            wandb.log({"gen_images": wandb.Image(make_grid(gen_images))}, commit=False)
            if ctxt_img is not None:
                wandb.log(
                    {"ctxt_images": wandb.Image(make_grid(ctxt_img))}, commit=False
                )

    # def on_validation_epoch_end(self) -> None:
    #     wandb.log({"FID": self.fid.compute()}, commit=False)
    #     self.fid.reset()

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
        ctxt_img: T.Tensor | None = None,
    ) -> T.Tensor:
        """Fully generate a batch of data from noise, given context
        information."""

        # Normalise the context
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            assert len(ctxt) == len(initial_noise)

        # Generate the sigma values
        sigmas = self.sigma_function(self.max_sigma, self.min_sigma)

        # Run the sampler
        outputs, _ = self.sampler_function(
            model=self,
            x=initial_noise,
            sigmas=sigmas,
            extra_args={"ctxt": ctxt, "ctxt_img": ctxt_img},
        )

        # Return the output
        return self.normaliser.reverse(outputs)

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.net.parameters())

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
