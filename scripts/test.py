import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from src.models.diffusion import ImageDiffusionGenerator

path = (
    "/home/users/l/leighm/scratch/Saved_Networks/image_processing/"
    "afhqv2_64_with_pos_enc/checkpoints/last.ckpt"
)
full_ckpt = ImageDiffusionGenerator.load_from_checkpoint(path)
net = full_ckpt.net

ema_net = full_ckpt.sigma_function.keywords["n_steps"]
sigma_encoder = full_ckpt.sigma_encoder
normaliser = full_ckpt.normaliser
ctxt_normaliser = getattr(full_ckpt, "ctxt_normaliser", None)


del full_ckpt
print(net)
