import torch
import torch.optim as optim
from model_definitions import VAE, Discriminator, combined_loss, create_synthetic_video, TransformerParamTanh
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Hyperparameters (adjust as needed) ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
VIDEO_HEIGHT = 128
VIDEO_WIDTH = 128
CHANNELS = 3
NUM_FRAMES = 16

# Transformer Hyperparameters
TRANSFORMER_IN_CHANNELS = NUM_FRAMES * CHANNELS
TRANSFORMER_NUM_HEADS = 12
TRANSFORMER_NUM_LAYERS = 12

# --- Data Loading ---
transform = transforms.Compose([
    transforms.Resize((VIDEO_HEIGHT, VIDEO_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

dataset = ImageFolder(root='data', transform=transform) # Assumes structure data/class1/img1.jpg, data/class1/img2.jpg, ...
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Loop ---
def train_transformer_tanh(device="cpu"):

    # --- Model Initialization ---
    tanh_layer = TransformerParamTanh(TRANSFORMER_IN_CHANNELS, TRANSFORMER_NUM_HEADS, TRANSFORMER_NUM_LAYERS, VIDEO_HEIGHT, VIDEO_WIDTH).to(device)
    vae = VAE().to(device)
    discriminator = Discriminator().to(device)

    # --- Optimizers ---
    optimizer_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_transformer = optim.Adam(tanh_layer.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(dataloader, leave=True)
        for idx, (real_images, _) in enumerate(loop): #data = images, _ = labels
            real_images = real_images.to(device)

            # --- Train Discriminator ---
            optimizer_discriminator.zero_grad()

            # Generate fake images
            generated_images, mu, log_var = vae(real_images, tanh_layer) #Forward with dynamic Tanh

            #Calculate discriminator Loss
            loss_discriminator = combined_loss(real_images, generated_images, discriminator, device)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # --- Train Generator (VAE) ---
            optimizer_vae.zero_grad()
            optimizer_transformer.zero_grad()

            # Generate fake images again. There could have been a change of the discriminator parameters on the previous run
            generated_images, mu, log_var = vae(real_images, tanh_layer) #Forward

            #Calculate generator Loss (reversed discriminator loss)
            loss_generator = -combined_loss(real_images, generated_images, discriminator, device) #Minimax loss
            loss_generator.backward()
            optimizer_vae.step()
            optimizer_transformer.step()

            # Print metrics to the terminal
            loop.set_postfix(loss_discriminator=loss_discriminator.item(), loss_generator=loss_generator.item())

    print(f"Training complete for TransformerParamTanh model")

# --- Main execution ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transformer_tanh(device)
