import math
from typing import Optional

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding (as used in many diffusion models).

    Args:
        t:   LongTensor of shape [B] with timestep indices.
        dim: embedding dimension.

    Returns:
        emb: FloatTensor of shape [B, dim]
    """
    half = dim // 2
    t = t.float()
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )  # [half]
    args = t[:, None] * freqs[None]  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*half]
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Block(nn.Module):
    """
    Basic building block used in the U-Net, with optional time embedding.

    For down blocks:
      - conv1 -> BN -> ReLU -> conv2 -> BN -> ReLU -> strided conv downsample

    For up blocks:
      - input is concat of skip and current => 2 * in_ch
      - conv1 -> BN -> ReLU -> conv2 -> BN -> ReLU -> ConvTranspose2d upsample
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: Optional[int] = None, up: bool = False) -> None:
        super().__init__()
        self.up = up

        if up:
            # up block: input will be concat of skip + current => 2 * in_ch
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            # down block
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.time_emb_dim = time_emb_dim
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [B, C, H, W]
            t_emb: [B, time_emb_dim] or None (projected timestep embedding)

        Returns:
            Tensor after convolutions and (up/down) sampling.
        """
        h = self.conv1(x)
        if hasattr(self, "time_mlp") and t_emb is not None:
            # Project t_emb to channel dim and add as a bias
            time_out = self.time_mlp(t_emb)  # [B, out_ch]
            h = h + time_out[..., None, None]  # broadcast over H, W

        h = self.relu(self.bnorm1(h))
        h = self.conv2(h)
        h = self.relu(self.bnorm2(h))

        return self.transform(h)


class SimpleUnet(nn.Module):
    """
    Time-conditioned U-Net.

    Input:
      - x_t: gamma-darkened image [B, 3, H, W]
      - t:   timestep indices [B]

    Output:
      - pred_x0: predicted bright image [B, 3, H, W]
    """

    def __init__(self, time_emb_dim: int = 256) -> None:
        super().__init__()
        image_channels = 3  # RGB
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3

        self.time_emb_dim = time_emb_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim=time_emb_dim, up=False)
            for i in range(len(down_channels) - 1)
        ])

        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim=time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]  (corrupted image x_t)
            t: [B] LongTensor (timestep indices)

        Returns:
            pred_x0: [B, 3, H, W] predicted bright images
        """
        # 1) Timestep embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)   # [B, time_emb_dim]
        t_emb = self.time_mlp(t_emb)                       # [B, time_emb_dim]

        # 2) U-Net forward with time conditioning
        x = self.conv0(x)
        residuals = []

        # Down path
        for down in self.downs:
            x = down(x, t_emb)
            residuals.append(x)

        # Up path
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t_emb)

        return self.output(x)
