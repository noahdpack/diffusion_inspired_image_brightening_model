# Diffusion-Inspired Image Brightening Model

The Diffusion-Inspired Image Brightening Model is an experimental computer vision project designed to brighten dark images using a time-conditioned U-Net. The project is inspired by the general structure of diffusion models, where an image is gradually transformed through a forward process and a neural network learns to reverse that transformation.

In this project, however, the forward process is not true diffusion noise. Instead of adding random Gaussian noise to an image, the model uses a deterministic gamma-based darkening process. A clean bright image is progressively darkened according to a timestep, and the model learns to reconstruct the original bright image from the darkened version.

The goal of this project was to explore whether a diffusion-style training setup could be adapted for an image enhancement task like brightening. More specifically, I wanted to test whether a model could learn to reverse different levels of synthetic darkness when given both the darkened image and the timestep used to create it.

## Project Structure

Here is the project file structure for reference:

```text
diffusion_image_brightening/
|-- src/
|   |-- data.py
|   |-- forward_pass.py
|   |-- model.py
|   |-- train.py
|   |-- utils.py
|   |-- visualize.py
|-- dataset/
|   |-- dark/
|   |-- bright/
|-- viz/
|-- gamma_denoiser.pth
```

The main files are:

* `data.py`: Handles the paired image dataset and dataloader.
* `forward_pass.py`: Defines the synthetic gamma-darkening process and reconstruction loss.
* `model.py`: Defines the time-conditioned U-Net model.
* `train.py`: Runs the training loop and saves model checkpoints.
* `visualize.py`: Loads a trained model and displays the original image, darkened image, and model reconstruction.
* `utils.py`: Contains helper functions for displaying tensor images.

## How the Model Works

The model is based on a simple idea:

1. Start with a clean bright image.
2. Apply a synthetic darkening process to make the image darker.
3. Give the darkened image and its timestep to the model.
4. Train the model to predict the original bright image.

The timestep matters because different timesteps correspond to different levels of darkening. Earlier timesteps apply little or no darkening, while later timesteps apply stronger gamma correction. This gives the model a way to learn how much correction it should apply.

The model is trained as a supervised reconstruction problem. It receives a darkened image `x_t` and timestep `t`, then predicts the original bright image `x_0`.

## Important Note About the Dataset

The model workflow doesn't contain actually use the images in the dark directory although a dark directory duel to the bright directory with corresponding file names is required to train the model. This is an artifact of previous projects versions that never got removed.

If you want to train a model and only have bright images, you can put your bright directory contents into the dark directory. Since the contents of the dark directory aren't used in anything but initilizing the data loader the images can be whatever as long as there is a corresponding dark directory image for each bright directory image with matching names.

## Forward Process

The forward process uses gamma-based dimming. The project defines 300 timesteps, with a gamma schedule that moves from 1.0 to 4.0.

At each timestep, the image is first mapped from the `[-1, 1]` tensor range back into `[0, 1]`. Then gamma correction is applied:

```python
x_t = x_0 ** gamma_t
```

After that, the image is mapped back into `[-1, 1]` for model training.

This process makes the image darker as the timestep increases. Since the process is deterministic, the same image and timestep will always produce the same darkened image.

It is important to note that this is not a probabilistic diffusion process. There is no random noise being added, and the model is not learning to denoise in the traditional diffusion sense. It is more accurate to describe this as a diffusion-inspired inverse gamma correction model.

## Dataset

The dataset is set up as paired image folders:

```text
dataset/
|-- dark/
|-- bright/
```

The `dark` and `bright` folders should contain matching filenames. The dataset class loads the corresponding dark and bright image pair, converts both images to RGB, applies the same spatial transforms to keep them aligned, and maps the image tensors into the `[-1, 1]` range.

Even though the dataloader returns both dark and bright images, the current experiment mainly uses the bright image as the clean target image. The synthetic gamma-darkened image is generated during training from the bright image.

This design leaves room for future versions of the project to use real dark / bright image pairs instead of only synthetic darkening.

## Model Architecture

The model is a time-conditioned U-Net.

The U-Net takes two inputs:

* `x_t`: the gamma-darkened image
* `t`: the timestep used to darken the image

The timestep is converted into a sinusoidal timestep embedding, similar to the kind used in diffusion models. That embedding is passed through a small MLP and injected into the U-Net blocks as a learned conditioning signal.

The U-Net has:

* An initial convolution layer
* A downsampling path
* An upsampling path
* Skip connections between matching down and up blocks
* A final convolution layer that predicts a 3-channel RGB image

The model outputs a prediction of the original bright image.

## Training

Training is handled by `train.py`.

For each batch:

1. Load a batch of bright images.
2. Randomly sample a timestep for each image.
3. Apply gamma dimming to create the darkened image.
4. Pass the darkened image and timestep into the model.
5. Compare the model prediction to the original bright image using mean squared error.
6. Update the model parameters with Adam.

The default training settings are:

```text
Image size: 64x64
Batch size: 128
Learning rate: 1e-3
Epochs: 75
Timesteps: 300
```

To train the model, run:

```bash
python -m src.train \
    --dark_dir ./dataset/dark \
    --bright_dir ./dataset/bright \
    --epochs 75 \
    --batch_size 128 \
    --img_size 64 \
    --lr 1e-3 \
    --checkpoint_path ./gamma_denoiser.pth
```

During training, the script prints the loss at each step and the average loss at the end of each epoch. It also saves a visualization after each epoch in the `viz/` folder so that progress can be inspected visually.

After training finishes, the final model weights are saved to the checkpoint path.

## Visualization

The visualization script shows three images:

1. The original bright image `x_0`
2. The gamma-darkened image `x_t`
3. The model prediction of the bright image

To visualize a trained model, run:

```bash
python -m src.visualize \
    --dark_dir ./dataset/dark \
    --bright_dir ./dataset/bright \
    --checkpoint ./gamma_denoiser.pth \
    --t_value 299
```

The `t_value` argument controls how much synthetic darkening is applied. A larger timestep means stronger darkening. Since the default number of timesteps is 300, `t_value 299` shows the model working from the strongest darkening level.

## What This Project Demonstrates

This project demonstrates several machine learning and computer vision concepts:

* Building a custom PyTorch dataset for paired image data
* Applying consistent transforms to paired images
* Creating a synthetic image corruption process
* Training a supervised image-to-image reconstruction model
* Using timestep embeddings for conditional image restoration
* Implementing a U-Net architecture in PyTorch
* Saving visualizations during model training
* Evaluating model behavior through reconstructed image outputs

## Limitations

The biggest limitation of the current project is that the forward process is deterministic. Since the model is trained to reverse gamma darkening rather than random noise, it is not a true diffusion model.

Another limitation is that the current training setup mostly uses the bright image and creates the darkened version synthetically. This makes the task more controlled, but it may not fully represent real low-light image enhancement. Real dark images can include noise, color shifts, blur, compression artifacts, and missing detail that cannot be captured by gamma correction alone.

The model also trains on relatively small 64x64 images by default. This makes experimentation faster, but higher-resolution results would require more compute and possibly architecture changes.

## Future Improvements

Future versions of this project could improve the system by:

* Training directly on real dark / bright image pairs
* Adding realistic low-light noise instead of only gamma darkening
* Comparing the model against simpler baselines like histogram equalization or direct gamma correction
* Adding quantitative metrics such as PSNR, SSIM, or perceptual loss
* Testing larger image sizes
* Experimenting with a true diffusion noising process
* Adding validation loss tracking and early stopping
* Improving the U-Net architecture with attention or residual blocks

## Set Up

### Prerequisites

Before getting started, make sure you have:

* Python 3.10+
* PyTorch
* torchvision
* NumPy
* Pillow
* matplotlib

A CUDA-capable GPU is recommended, but the project can run on CPU for small experiments.

### Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd diffusion_image_brightening
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Or on Windows activate with:

```bash
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure your dataset is organized like this:

```text
dataset/
|-- dark/
|-- bright/
```

Then run training:

```bash
python -m src.train
```

After training, run visualization:

```bash
python -m src.visualize --checkpoint ./gamma_denoiser.pth
```

## Final Note

This project started as an attempt to apply the intuition behind diffusion models to a practical image enhancement task. The result is not a full diffusion model, but it is a useful experiment in time-conditioned image reconstruction. It shows how ideas from diffusion-style modeling can be adapted, simplified, and tested in a more controlled supervised learning setting.
