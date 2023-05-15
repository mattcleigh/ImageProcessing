import fnmatch
from pathlib import Path

import wandb
from PIL import Image
from tqdm import tqdm

api = wandb.Api()
run = api.run("mleigh/image_processing/runs/tayfnmiu")
file_wilcards = "*images_*_*.png"
image_path = "/home/users/l/leighm/ImageProcessing/"
prefix = "media/images/"

for file in tqdm(run.files()):
    if fnmatch.fnmatch(file.name, file_wilcards):
        file.download(image_path, exist_ok=True)


def images_to_gif(image_fnames, fname):
    image_fnames.sort(key=lambda x: int(x.name.split("_")[-2]))  # sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(
        f"{fname}.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=1,
        loop=0,
    )


images_to_gif(list(Path(image_path + prefix).glob(file_wilcards)), "test")
