# TransformerParamTanh
A generative vid# Generative Video Creation with TransformerParamTanh

## Architecture Overview

(Placeholder Image: A diagram showing the VAE architecture, with a TransformerParamTanh layer inserted after the decoder. Include the Transformer processing the video and feeding parameters into the Tanh layer. Include the Discriminator network.)

The architecture consists of a Variational Autoencoder (VAE) acting as the generator and a Discriminator network. The VAE is trained to generate realistic images, while the Discriminator tries to distinguish between real and generated images. A `TransformerParamTanh` layer is inserted after the decoder to introduce a dynamic normalization effect, controlled by a Transformer.

## Network Details

### VAE (Generator)

*   **Architecture:** Encoder (Conv2D, ReLu, MaxPool2D) -> Latent Space (Linear layers for mu/logvar) -> Decoder (ConvTranspose2D, ReLu, Sigmoid).  See `model_definitions.py` for specific layer details.
*   **Input:** Noise vector.
*   **Output:** Generated image (B, C, H, W).

### Discriminator

*   **Architecture:** (The layers are symmetric with the encoder.) See `model_definitions.py` for layer details.
*   **Input:** Real or generated image (B, C, H, W).
*   **Output:** Probability of the input being a real image.

### TransformerParamTanh Layer

*   **Type:** `TransformerParamTanh`
*   **Transformer Architecture:** Transformer Encoder with a Linear layer to predict alpha parameters.
*   **Transformer Parameters:**
    *   Number of Heads: 12.
    *   Number of Layers: 12.
    *   Input Channels:  TRANSFORMER_IN_CHANNELS (NUM_FRAMES*CHANNELS)
*   **Alpha Derivation:** The Transformer processes a *synthetic video* created from the generated image and predicts a matrix of `alpha` values for the Tanh layer. The synthetic video is created by stacking slightly modified versions of the generated image.

## Training Data

*   Images of shape (B, C, H, W) = (16, 3, 128, 128).
*   The `ImageFolder` dataset from `torchvision` is used for loading images.

## Training Procedure

1.  A synthetic video is created from the generated image to feed into the Transformer.
2.  The Discriminator is trained to distinguish between real and generated images using the combined Wasserstein loss with gradient penalty.
3.  The VAE is trained to generate realistic images that can fool the Discriminator, also using the combined Wasserstein loss.
4.  The Transformer and `TransformerParamTanh` layer's parameters are learned during the VAE training process.

## Usage

1.  Ensure you have PyTorch, torchvision, and tqdm installed.
2.  Organize your image data into a directory structure compatible with `ImageFolder`.
3.  Run the `training.py` script to train the model.eo model with a feedback loop from a second network based on transformer
