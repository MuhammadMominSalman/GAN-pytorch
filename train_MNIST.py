import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import DCGAN
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
# Hyperparameters
latent_size = 64
batch_size = 100
learning_rate = 0.0002
epochs = 100
log_dir = './runs/gan_mnist'
device = "cuda"

# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Models
generator = DCGAN.Generator(latent_size).to(device)
discriminator = DCGAN.Discriminator().to(device)



# Loss and optimizer
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# TensorBoard summary writer
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# Fixed noise for consistent image generation
fixed_noise = torch.randn(batch_size, latent_size).to(device)

# Training loop
for epoch in range(epochs):
    for i, (images, _) in enumerate(dataloader):
        # Flatten images
        images = images.to(device)

        # Create labels
        real_labels = torch.ones_like(images)
        fake_labels = torch.zeros_like(images)

        # Train discriminator
        optimizer_d.zero_grad()
        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

    # Log losses to TensorBoard
    d_loss = d_loss_real + d_loss_fake
    writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
    writer.add_scalar('Loss/Generator', g_loss.item(), epoch)

    # Log losses to TensorBoard
    d_loss = d_loss_real + d_loss_fake
    writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
    writer.add_scalar('Loss/Generator', g_loss.item(), epoch)

    # Log generated images to TensorBoard
    if epoch % 5 == 0:  # Log images every 10 epochs
        with torch.no_grad():
            fake_images = generator(fixed_noise).view(-1, 1, 28, 28)  # Reshape to (batch_size, channels, height, width)
            img_grid = vutils.make_grid(fake_images, normalize=True)
            writer.add_image('Generated Images', img_grid, epoch)


    print(f'Epoch [{epoch}/{epochs}], d_loss: {d_loss_real + d_loss_fake}, g_loss: {g_loss}')

# Close the writer after training
writer.close()