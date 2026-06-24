"""Image conversion helpers for ComfyUI IMAGE tensors."""

import numpy as np
import torch


def pil_images_to_comfy(images):
    """Convert a PIL image or list of PIL images to [B, H, W, C] float32 tensors."""
    if not isinstance(images, (list, tuple)):
        images = [images]

    tensors = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(array))

    return torch.stack(tensors, dim=0).contiguous()

