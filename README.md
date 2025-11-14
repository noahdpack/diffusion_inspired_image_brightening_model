# Time-Conditioned Inverse Gamma Model (Diffusion-Style U-Net)

This project explores a **diffusion-style, time-conditioned U-Net** trained to undo a synthetic gamma-based darkening process on images.

Instead of implementing a full probabilistic diffusion model with noisy forward and reverse Markov chains, this project uses a **deterministic "forward" operator** based on gamma correction. The model learns to approximate the inverse of this operator for different gamma strengths.

The result is a clean, self-contained experiment that shows how diffusion-style architectures behave when stochastic noise is removed: they collapse into a **deterministic, parametric tone-mapping network**.

---

## 🔍 Motivation

I wanted to understand:

- How time-conditioned U-Nets used in diffusion models work in practice.
- What happens when you **remove noise and sampling** from the diffusion framework.
- Whether such a network would learn something interesting on a simple, synthetic task.

To do this, I:

1. Took paired images (dark, bright).
2. Treated the **bright** image as the ground truth `x₀`.
3. Defined a synthetic "forward" process that progressively **darkens** `x₀` using a gamma schedule.
4. Trained a U-Net with sinusoidal timestep embeddings to reconstruct `x₀` from its darkened version `x_t` and the timestep index `t`.

---

## 🧠 What the model actually does

- The forward process is:

  \[
  x_t = \text{gamma}(x_0, \gamma_t),
  \]

  where `γ_t` increases from 1.0 to 4.0 over `T = 300` timesteps.

- There is **no Gaussian noise** and no Markov chain over time.
- The model sees `(x_t, t)` and learns to predict `x₀` in **one pass**.

In other words, this is **not a full diffusion model**.  
It is better described as a:

> **Time-conditioned inverse gamma network**  
> or  
> **Neural approximation of a family of inverse tone-mapping curves**,  
> implemented with a diffusion-style architecture (U-Net + timestep embeddings).

This was intentional: the goal was to **study the behavior** of the architecture in a fully deterministic setting, not to build a production image brightener.

---

## 🧱 Architecture

Core components (all in `src/`):

- `data.py`
  - `PairedImageDataset` – loads `(dark, bright)` image pairs from folders.
  - `PairedTransform` – applies the same random augmentation to both images.
  - `build_dataloader(...)` – returns a `DataLoader` with images in `[-1, 1]`.

- `gamma_forward.py`
  - `T = 300` – number of synthetic timesteps.
  - `gamma_schedule` – linearly spaced `γ ∈ [1, 4]`.
  - `apply_gamma_dimming(x0, t, device)` – deterministic gamma darkening of `x0`.
  - `get_loss(model, x0, t, device)` – computes `MSE(model(x_t, t), x0)`.

- `model.py`
  - `timestep_embedding(t, dim)` – sinusoidal timestep encoding.
  - `Block` – basic building block with optional time embedding.
  - `SimpleUnet` – diffusion-style U-Net conditioned on timestep indices.

- `utils.py`
  - `show_tensor_image(image)` – helper for visualizing images in `[-1, 1]`.

- `train.py`
  - End-to-end training script that:
    - Builds a dataloader.
    - Samples random timesteps `t`.
    - Applies `apply_gamma_dimming` to obtain `x_t`.
    - Trains `SimpleUnet` to reconstruct `x₀`.

- `visualize.py`
  - `predict_and_plot_image(...)` – shows:
    - original bright image `x₀`,
    - gamma-darkened `x_t`,
    - model prediction `\hat{x₀}`.
  - Can be run as a separate script from the command line.

---

## 📂 Data

The dataset is expected to live under:

```text
dataset/
├─ dark/
│  ├─ img_001.png
│  ├─ ...
└─ bright/
   ├─ img_001.png
   ├─ ...
