from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def read_rgb(path, image_size=None):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if image_size:

            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    Image.fromarray(array).save(path)
