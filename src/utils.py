import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch import Tensor


def show_tensor_image(image: Tensor) -> None:
    """
    Utility to visualize a tensor image stored in [-1, 1] range.

    Accepts either [C, H, W] or [B, C, H, W] (in which case it uses the first image).
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2.0),     # [-1,1] -> [0,1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW -> HWC
        transforms.Lambda(lambda t: t * 255.0),
        transforms.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if image.dim() == 4:
        image = image[0, :, :, :]

    pil_img = reverse_transforms(image)
    plt.imshow(pil_img)
    plt.axis("off")
