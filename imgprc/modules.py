from copy import deepcopy

import numpy as np
from dotmap import DotMap

import torch as T
import torch.nn as nn

from mattstools.cnns import avg_pool_nd, ResNetBlock


class PatchDiscriminator(nn.Module):
    """A discriminator which uses ResNetBlocks to process a decision on image patches

    Unlike a standard discriminator which would eventually flatten the outputs and
    pass through a dense network, this module only uses ResNet blocks and ouptus a full
    (lower) resolution image, where each of the output pixels represents overlapping
    patches of the input.

    The final convoluation operation has a single output channel, the decision of if a
    patch is real or fake.

    The discriminator therefore focusses on texture and colour rather than the whole
    image itself and is better suites for upscaling tasks when combined with some
    supervised reconstruction loss.
    """

    def __init__(
        self,
        inpt_size: list,
        inpt_channels: int,
        outp_size: int = 16,
        ctxt_dim: int = 0,
        start_channels: int = 8,
        max_channels: int = 64,
        resnet_kwargs: DotMap = None,
    ) -> None:
        """
        args:
            inpt_size: Spacial dimensions of the inputs
            inpt_channels: Number of channels in the inputs
            outp_size: Size of the output image
                inpt_size/outp_size determines the number of layers and the receptive
                size of the patches
            ctxt_dim: Size of the contextual tensor
            resnet_kwargs: Kwargs for the ResNetBlocks
        """
        super().__init__()

        ## Safe dict defaults
        resnet_kwargs = resnet_kwargs or DotMap()

        ## Class attributes
        self.inpt_size = inpt_size
        self.inpt_channels = inpt_channels
        self.ctxt_dim = ctxt_dim
        self.outp_size = outp_size
        self.start_channels = start_channels
        self.max_channels = max_channels
        self.dims = len(inpt_size)

        ## The downsampling layer (not learnable)
        stride = 2 if self.dims != 3 else (2, 2, 2)
        self.down_sample = avg_pool_nd(self.dims, kernel_size=stride, stride=stride)

        ## The first ResNet block changes to the starting channel dimension
        first_kwargs = deepcopy(resnet_kwargs)
        first_kwargs.nrm_groups = 1
        self.first_block = ResNetBlock(
            inpt_channels=inpt_channels,
            ctxt_dim=ctxt_dim,
            outp_channels=start_channels,
            **first_kwargs,
        )

        ## Keep track of the spacial dimensions for each input and output layer
        inp_size = np.array(inpt_size)
        inp_c = start_channels
        out_c = start_channels * 2

        ## Start creating the levels (should exit but max 100 for safety)
        resnet_blocks = []
        for lvl in range(100):

            ## Add the resnet block
            resnet_blocks.append(
                ResNetBlock(
                    inpt_channels=inp_c,
                    ctxt_dim=ctxt_dim,
                    outp_channels=out_c,
                    **resnet_kwargs,
                )
            )

            ## Exit if the next iteration would lead too small spacial dimensions
            if min(inp_size) // 2 <= outp_size:
                break

            ## Update the dimensions for the next iteration
            inp_size = inp_size // 2  # Halve the spacial dimensions
            inp_c = out_c
            out_c = min(out_c * 2, max_channels)  # Double the channels

        ## Combine layers into a module list
        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        ## The final output, must have last layer be 1 part of the UNet
        final_kwargs = deepcopy(resnet_kwargs)
        final_kwargs.nrm_groups = 1
        final_kwargs.drp = 0 # No dropout! These are the outputs!
        self.final_block = ResNetBlock(
            inpt_channels=out_c,
            ctxt_dim=ctxt_dim,
            outp_channels=1,
            **final_kwargs,
        )

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor = None):
        """Forward pass of the network"""

        ## Pass through the first convolution layer to embed the channel dimension
        inpt = self.first_block(inpt)

        ## Pass through the ResNetBlocks and the downsampling
        for layer in self.resnet_blocks:
            inpt = layer(inpt, ctxt)
            inpt = self.down_sample(inpt)

        ## Pass through the final block and return
        return self.final_block(inpt, ctxt)


if __name__ == "__main__":
    test = PatchGANDiscriminator(
        [128, 128],
        3,
        32,
        ctxt_dim=0,
    )
    test(T.randn((1, 3, 128, 128))).shape
