# Team 0
# ImageColorization

This project implements a custom Deep Learning pipeline to automatically colorize grayscale landscape images. It utilizes a custom-built U-Net generator and a PatchGAN discriminator to transition from basic structural colorization to vibrant, realistic, and context-aware image generation.

## Architecture Highlights

The system is built around a Conditional Generative Adversarial Network (cGAN) architecture, heavily inspired by Pix2Pix, but tailored with custom components:

### 1. The Generator (Custom U-Net)
Operates entirely in the LAB color space, taking the `L` (lightness) channel as input to predict the `ab` (color) channels.
* **Custom ResNet Encoder:** Built from scratch using residual `BasicBlock` structures to extract deep spatial features while preventing vanishing gradients. It feeds precise skip connections (`e0` to `e4`) directly into the decoder.
* **U-Net Decoder:** Leverages the encoder's skip connections to progressively upsample the feature maps, ensuring that fine, high-resolution details are perfectly preserved in the final colorized output.

### 2. The Discriminator (PatchGAN Discriminator)
Instead of outputting a single "Real or Fake" value for the entire image, the discriminator is designed as a PatchGAN.
* It penalizes structure at the scale of local patches.
* This forces the generator to learn high-frequency, realistic local textures (like the grain of wood or individual blades of grass) rather than just broad, blurry color blobs.

### 3. Two-Phase Optimization
* **Phase 1 (Baseline):** The generator is pre-trained using strictly L1 Loss. This establishes structural accuracy and spatial awareness (e.g., sky is up, ground is down) but typically results in "safe," washed-out colors.
* **Phase 2 (Adversarial):** The discriminator is introduced using a Least Squares GAN approach (MSE Loss). This forces the generator to take risks, injecting vibrant and photorealistic colors to fool the discriminator.

## Dataset
Trained on the [Landscape Image Colorization Dataset](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization) from Kaggle.
* **Total Images:** 7,129
* **Training Split:** 80%

## Setup & Installation
Install the required dependencies before running the project:

```bash
pip install -r requirements.txt
```
*Note: The global project configuration (paths, hyperparameters) is managed in `src/setup/config.py`.*

## Training the Model
The training pipeline is modular and split into executable scripts located in `src/training/`.

### 1. Hardware Benchmarking
Run `src/training/benchmark.py` first. It profiles your specific machine to find the optimal hardware parameters (like batch size and DataLoader workers) to maximize GPU utilization.

### 2. Baseline Training
`src/training/training_pipeline.py` trains the baseline U-Net generator using L1 loss. 
The model learns basic color mapping. Trained model checkpoints are saved every 5 epochs to the `trained_models/` directory.

### 3. GAN Training
`src/training/train_gan.py` loads the baseline weights and introduces the discriminator for the adversarial training phase. 
Colors become significantly more vibrant and realistic. Both generator and discriminator checkpoints are saved every 5 epochs to the `trained_models/` directory.

## Inference (Colorizing Images)
Use `src/colorize_image.py` to test your trained model and generate colorized images.

You will need to specify the model and provide an optional path to a target image. If no path is specified, it will colorize a random image from the dataset.
* **Output:** All colorized results are saved to the `predicted_data/` directory.
