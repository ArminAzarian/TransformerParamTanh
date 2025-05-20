import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Utility functions ---
def reshape_video_to_channels(video_sequence):
  """
  Reshapes a video sequence (B, T, C, H, W) into (B, T*C, H, W).
  Useful for processing with 2D CNNs or Transformers.
  """
  B, T, C, H, W = video_sequence.shape
  return video_sequence.reshape(B, T * C, H, W)

def create_synthetic_video(image, num_frames=16):
    """
    Creates a synthetic video from a single image by stacking slightly modified versions.

    Args:
        image: A tensor of shape (B, C, H, W).
        num_frames: The number of frames in the synthetic video.

    Returns:
        A tensor of shape (B, T, C, H, W).
    """
    B, C, H, W = image.shape
    video = torch.zeros((B, num_frames, C, H, W), device=image.device)
    for t in range(num_frames):
        # Add slight noise to each frame
        noise = torch.randn_like(image) * 0.01 * t  # Scale noise with frame index
        video[:, t] = torch.clamp(image + noise, 0, 1) #Clamp between 0 and 1
    return video

# 1.4 Transformer-based Parametric Dynamic Tanh
class TransformerParamTanh(nn.Module):
    def __init__(self, in_channels, num_heads, num_layers, video_height, video_width):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels * video_height * video_width, nhead=num_heads),
            num_layers=num_layers
        )
        self.parameter_predictor = nn.Linear(in_channels * video_height * video_width, video_height * video_width)
        self.video_height = video_height
        self.video_width = video_width
        #Initialize weights
        nn.init.xavier_uniform_(self.parameter_predictor.weight)


    def forward(self, x, video):
        # Flatten the video frames
        video_reshape = reshape_video_to_channels(video)

        # Flatten spatial dimensions and treat channels as features
        B, C, H, W = video_reshape.shape
        video_flattened = video_reshape.reshape(B, C * H * W)

        # Add a sequence dimension of length 1
        video_flattened = video_flattened.unsqueeze(1)

        # Pass the video through the Transformer
        transformer_output = self.transformer_encoder(video_flattened)

        # Predict the parameters
        alpha = self.parameter_predictor(transformer_output.squeeze(1))

        # Reshape alpha to have the shape of an image
        B = x.shape[0]
        alpha = alpha.reshape(B, 1, self.video_height, self.video_width)
        return torch.tanh(alpha * x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 48, kernel_size=3, padding="same") #Explicit padding with "same" will adapt the input dimension
        self.enc2 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.enc3 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#For asymmetric stride
        #Latent space parameterization layers
        self.fc1 = nn.Linear(48 * 16 * 16, 64)  #  mean
        self.fc2 = nn.Linear(48 * 16 * 16, 64)  #  log variance
        self.fc3 = nn.Linear(64, 48 * 16 * 16) # from latent space to decoder

        # Decoder
        self.dec1 = nn.ConvTranspose2d(48, 48, kernel_size=3, padding="same")
        self.dec2 = nn.ConvTranspose2d(48, 48, kernel_size=3, padding="same")
        self.dec3 = nn.ConvTranspose2d(48, 3, kernel_size=3, padding="same")


    def encoder(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x)) #Sigmoid, so all the outputs are compressed from 0 to 1
        return x

    def forward(self, x, tanh_layer = None):
        #Encode
        x_encoded = self.encoder(x)
        #Latent space parameterization
        mu = F.relu(self.fc1(x_encoded))
        log_var = F.relu(self.fc2(x_encoded))
        #Sampling
        z = self.reparameterize(mu, log_var)
        #Mapping to the Decoder
        x_reconstructed = F.relu(self.fc3(z))
        x_reconstructed = x_reconstructed.reshape(-1, 48, 16, 16) # From linear to image
        #Decode
        x_reconstructed = self.decoder(x_reconstructed)
        #Dynamic Tanh
        if tanh_layer is not None:
            #We have a single image, we will create a video to feed it to the Transformer
            video = create_synthetic_video(x_reconstructed)
            x_reconstructed = tanh_layer(x_reconstructed, video)
        return x_reconstructed, mu, log_var #Return also mu and log_var for the loss function calculation

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 48, kernel_size=3, padding="same") #Explicit padding with "same" will adapt the input dimension
        self.enc2 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.enc3 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#For asymmetric stride
        #Latent space parameterization layers
        self.fc1 = nn.Linear(48 * 16 * 16, 64)  #  mean
        self.fc2 = nn.Linear(48 * 16 * 16, 64)  #  log variance
        self.fc3 = nn.Linear(64, 1) # from latent space to decoder

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # From linear to image
        return x

# --- Loss Functions ---

def wasserstein_loss(real_samples, generated_samples, discriminator):
    # Calculate discriminator output for real and generated samples
    real_validity = discriminator(real_samples)
    fake_validity = discriminator(generated_samples)

    # Calculate Wasserstein loss
    wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
    return wasserstein_distance

def gradient_penalty(real_samples, generated_samples, discriminator, device="cpu"):
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.randn((real_samples.size(0), 1, 1, 1), device = device)
    # Get random interpolation between real and fake samples
    interpolated = (alpha * real_samples + ((1 - alpha) * generated_samples)).requires_grad_(True)
    interpolated.to(device)
    interpolated_output = discriminator(interpolated)

    grad_outputs = torch.ones(interpolated_output.size(), dtype=torch.float, device=device, requires_grad=False)

    # Get gradient of output w.r.t. interpolated input
    grad_interpolated = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_interpolated = grad_interpolated.view(real_samples.size(0), -1)
    grad_norm = grad_interpolated.norm(2, dim=1)
    grad_penalty = torch.mean((grad_norm - 1) ** 2)
    return grad_penalty

#Combined Loss = Wasserstein loss + gradient_penalty
def combined_loss(real_samples, generated_samples, discriminator, device, lambda_gp = 10):
    wasserstein = wasserstein_loss(real_samples, generated_samples, discriminator)
    grad_pen = gradient_penalty(real_samples, generated_samples, discriminator, device)
    return wasserstein + lambda_gp * grad_pen
