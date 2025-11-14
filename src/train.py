"""
Training script for the time-conditioned inverse gamma model.

Example:

    python -m src.train \
        --dark_dir ./dataset/dark \
        --bright_dir ./dataset/bright \
        --epochs 75 \
        --batch_size 128 \
        --img_size 64 \
        --lr 1e-3 \
        --checkpoint_path ./gamma_denoiser.pth
"""

import argparse
from pathlib import Path

import torch
from torch.optim import Adam

from .data import build_dataloader
from .forward_pass import T, get_loss
from .model import SimpleUnet
from .visualize import predict_and_plot_image


def train(
    dark_dir: str,
    bright_dir: str,
    img_size: int = 64,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 75,
    checkpoint_path: str = "gamma_denoiser.pth",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataloader = build_dataloader(
        dark_dir=dark_dir,
        bright_dir=bright_dir,
        img_size=img_size,
        batch_size=batch_size,
    )

    model = SimpleUnet().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    print("Starting training...")
    global_step = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()

        for step, (_, bright_image) in enumerate(dataloader):
            optimizer.zero_grad()

            # bright_image is our clean x0
            bright_image = bright_image.to(device)

            # Sample random timesteps for each image in the batch
            bsz = bright_image.size(0)
            t = torch.randint(low=0, high=T, size=(bsz,), device=device).long()

            # Compute loss: model learns to reconstruct x0 from (x_t, t)
            loss = get_loss(model, bright_image, t, device)
            loss.backward()
            optimizer.step()

            global_step += 1
            if (step + 1) % 10 == 0:
                print(f"[Epoch {epoch + 1} | Step {step + 1}] Loss: {loss.item():.4f}")

        # Visual check after each epoch (uses a fresh batch from dataloader)
        try:
            predict_and_plot_image(model, dataloader, device, t_value=T - 1)
        except Exception as e:
            print(f"Visualization failed (this can happen on headless servers): {e}")

    # Save final model
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark_dir", type=str, default="./dataset/dark")
    parser.add_argument("--bright_dir", type=str, default="./dataset/bright")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--checkpoint_path", type=str, default="gamma_denoiser.pth")
    args = parser.parse_args()

    train(
        dark_dir=args.dark_dir,
        bright_dir=args.bright_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
