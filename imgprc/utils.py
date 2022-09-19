"""
A collection of misculaneous classes and functions useful for the training process
"""

import argparse
from typing import Tuple

from mattstools.utils import get_standard_configs, args_into_conf, str2bool


def get_configs(**kwargs) -> Tuple[dict, dict, dict]:
    """Loads, modifies, and returns three configuration dictionaries using command
    line arguments for a graph discriminator
    """

    data_conf, net_conf, train_conf = get_standard_configs(**kwargs)

    # parser = argparse.ArgumentParser(allow_abbrev=False)
    # parser.add_argument("--batch_size", type=int, help="Number of samples per batch")
    # parser.add_argument(
    # "--drop_last", type=str2bool, help="Drop final incomplete batch"
    # )
    # args, _ = parser.parse_known_args()

    ## Load the arguments
    # args_into_conf(args, data_conf, "batch_size")
    # args_into_conf(args, data_conf, "drop_last")

    return data_conf, net_conf, train_conf
