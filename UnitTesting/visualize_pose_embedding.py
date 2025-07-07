import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.pose_embedding import SinusoidalPosEmb  # Import Pose Embedding module

# 📌 Define Pose Embedding Model
dim = 16  # Choose an embedding size
model = SinusoidalPosEmb(dim)

# 📌 Generate a range of time steps
x_values = torch.linspace(0, 100, steps=100).view(-1, 1)  # Ensure correct shape (100, 1)

# 📌 Compute embeddings and remove extra dimension
with torch.no_grad():
    embeddings = model(x_values).squeeze(1).numpy()  # Now shape is (100, 16)

print("Fixed Embedding Shape:", embeddings.shape)  # Should be (100, dim)

# 📌 Plot Sinusoidal Embeddings
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Top Plot: First 8 Dimensions (sin components)
for i in range(dim // 2):
    ax[0].plot(x_values.numpy().flatten(), embeddings[:, i], label=f"Dim {i+1}")

ax[0].set_title("Sinusoidal Embeddings (sin components)")
ax[0].set_xlabel("Time Step (x)")
ax[0].set_ylabel("Embedding Value")
ax[0].legend()
ax[0].grid()

# Bottom Plot: Last 8 Dimensions (cos components)
for i in range(dim // 2, dim):
    ax[1].plot(x_values.numpy().flatten(), embeddings[:, i], label=f"Dim {i+1}")

ax[1].set_title("Sinusoidal Embeddings (cos components)")
ax[1].set_xlabel("Time Step (x)")
ax[1].set_ylabel("Embedding Value")
ax[1].legend()
ax[1].grid()

plt.suptitle(f"Visualization of Sinusoidal Positional Embeddings (Dim={dim})")
plt.show()