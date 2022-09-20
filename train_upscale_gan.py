"""
Main callable script to train the flavour tagger
"""

from pathlib import Path
from torch.utils.data import DataLoader

from imgprc.datasets import ImageDataset
from imgprc.utils import get_configs

from mattstools.trainer import Trainer
from imgprc.networks import UNetSRGAN

import wandb


def main():
    """Run the script"""

    ## Collect the session arguments, returning the configurations for the session
    data_conf, net_conf, train_conf = get_configs(
        def_net="config/unet_gan.yaml", def_train="config/train_gan.yaml"
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
    valid_loader = DataLoader(valid_set, **train_conf.loader_kwargs, shuffle=False)
    train_loader = DataLoader(train_set, **train_conf.loader_kwargs, shuffle=True)

    ## Use the shape of the data to define the network's IO
    net_conf.base_kwargs.inpt_dim = valid_set.image_shape()
    net_conf.base_kwargs.outp_dim = valid_set.image_shape()
    net_conf.base_kwargs.ctxt_dim = 0  # train_set.num_classes()
    net_conf.steps_per_epoch = len(train_loader)

    ## Create the network
    network = UNetSRGAN(**net_conf)
    print(network)

    ## Create the save folder for the network and store the configuration dicts
    network.save_configs(data_conf, net_conf, train_conf)

    ## Create the trainer and run the loop
    trainer = Trainer(network, train_loader, valid_loader, **train_conf.trainer_kwargs)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
