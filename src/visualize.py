"""
Standalone visualization script.

Usage (example):

    python -m src.visualize \
        --dark_dir ./dataset/dark \
        --bright_dir ./dataset/bright \
        --checkpoint ./gamma_denoiser.pth \
        --t_value 299
"""

import argparse

import torch

from .data import build_dataloader
from .forward_pass import T, apply_gamma_dimming
from .model import SimpleUnet
from .utils import show_tensor_image
import matplotlib.pyplot as plt


def predict_and_plot_image(model, dataloader, device: str, t_value: int | None = None) -> None:
    """
    Visualize one clean bright image x0, its gamma-darkened version x_t at timestep t,
    and the model's prediction hat{x0}.
    """
    model.eval()

    # Grab one batch from dataloader
    _, bright_batch = next(iter(dataloader))  # ignore dark_img
    x0 = bright_batch[0:1].to(device)         # [1, 3, H, W]

    # Choose timestep
    if t_value is None:
        t_value = T - 1  # max darkening

    t = torch.tensor([t_value], dtype=torch.long, device=device)
    x_t = apply_gamma_dimming(x0, t, device=device)

    # Model prediction
    with torch.no_grad():
        pred_x0 = model(x_t, t)

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    show_tensor_image(x0.cpu())
    plt.title("Clean x₀ (bright)")

    plt.subplot(1, 3, 2)
    show_tensor_image(x_t.cpu())
    plt.title(f"Gamma-darkened x_t (t={t_value})")

    plt.subplot(1, 3, 3)
    show_tensor_image(pred_x0.cpu())
    plt.title("Model prediction of x₀")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark_dir", type=str, default="./dataset/dark")
    parser.add_argument("--bright_dir", type=str, default="./dataset/bright")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--t_value", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = build_dataloader(
        dark_dir=args.dark_dir,
        bright_dir=args.bright_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    model = SimpleUnet().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    predict_and_plot_image(model, dataloader, device, t_value=args.t_value)


if __name__ == "__main__":
    main()
