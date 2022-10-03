"""
Main callable script to train the flavour tagger
"""

from torch.utils.data import DataLoader

from mattstools.trainer import Trainer

from imgprc.utils import get_configs
from imgprc.datasets import ImageDataset
from imgprc.diffusion import DiffusionSuperResolution

import wandb


def main():
    """Run the script"""

    ## Collect the session arguments, returning the configurations for the session
    data_conf, net_conf, train_conf = get_configs(
        def_net="config/unet_diff.yaml", def_train="config/train.yaml"
    )

    ## Start a weights and biases session
    wandb.init(
        entity="mleigh",
        project="ImageUpscaling",
        name=net_conf.base_kwargs.name,
        resume=train_conf.trainer_kwargs.resume,
        id=train_conf.wandb_id or wandb.util.generate_id(),
    )
    train_conf.wandb_id = wandb.run.id

    ## Get the training set and validation set, as well as the backward transform
    train_set = ImageDataset(is_train=True, **data_conf)
    valid_set = ImageDataset(is_train=False, **data_conf)

    ## Make the data loaders
    train_loader = DataLoader(train_set, **train_conf.loader_kwargs, shuffle=True)
    valid_loader = DataLoader(valid_set, **train_conf.loader_kwargs, shuffle=False)

    ## Use the shape of the data to define the network's IO
    net_conf.base_kwargs.inpt_dim = valid_set.image_shape()
    net_conf.base_kwargs.outp_dim = valid_set.image_shape()
    net_conf.base_kwargs.ctxt_dim = 0

    ## Create the network
    network = DiffusionSuperResolution(**net_conf)
    print(network)

    ## Create the save folder for the network and store the configuration dicts
    network.save_configs(data_conf, net_conf, train_conf)

    ## Create the trainer and run the loop
    trainer = Trainer(network, train_loader, valid_loader, **train_conf.trainer_kwargs)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
