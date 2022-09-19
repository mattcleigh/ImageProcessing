"""
Functions required for plotting and saving images
"""

from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

import torch as T
import torchvision as tv

from mattstools.torch_utils import to_np


def save_images(out_path: Path, input: T.Tensor):

    ## Make the data a grid
    tv.utils.make_grid(input)

    ## Convert to a numpy array
    if isinstance(input, T.Tensor):
        input = to_np(input)

    ## Data is typically in C,X,Y, plotting must be in X,Y,C
    plt.imshow(np.transpose(input, (1, 2, 0)))

    ## Save the output
    plt.savefig(out_path)
