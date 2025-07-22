'''
Validation Code for action : Velocity (V-omg)

'''

import torch
import torchvision.transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import os

#Import all the modules here
import sys
sys.path.insert(0,"C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/")

# Import modules
from modules.resnet import get_resnet50
from modules.resnet import get_resnet18
from modules.dataset import CustomDataset
from modules.unet2 import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(checkpoint_path="trained_policyV1.pth"):
    # Define architecture
    vision_encoder = get_resnet18().to(device)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        local_cond_dim=16,
        global_cond_dim=5160, #(if Resnet50 : 2048*25 + 1*25 + 1*25 = 51250) (if Resnet18 : 512*25 + 1*25 + 1*25 = 12850)
        #global_cond_dim=2580,
        diffusion_step_embed_dim=256,
        #down_dims=[256, 512, 1024],
        down_dims=[128, 256, 512],
        kernel_size=3,
        n_groups=8).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    noise_pred_net.load_state_dict(checkpoint["model_state_dict"])
    vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
    vision_encoder.eval()
    noise_pred_net.eval()

    return vision_encoder, noise_pred_net

# Validation logic
def validate(dataloader, vision_encoder, noise_pred_net, num_steps=100,
             video_path="denoising_trajectory.mp4", image_path="input_image.png"):

    # Use config access to avoid deprecation warnings
    diffusion_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2"
    )

    batch = next(iter(dataloader))
    idx = np.random.randint(0, batch["images"].shape[0])

    # Get sample
    images = batch["images"][idx].unsqueeze(0).to(device)         # [1, 25, 3, 96, 96]
    imuV = batch["imu_v"][idx].unsqueeze(0).to(device).float()    # [1, 25]
    imuOmg = batch["imu_omg"][idx].unsqueeze(0).to(device).float()
    posX = batch["posX"][idx].unsqueeze(0).to(device).float()
    posY = batch["posY"][idx].unsqueeze(0).to(device).float()
    true_velocities = batch["actions"][idx].cpu().numpy()         # [100, 2] = (V, ω)

    # Save input image (first frame)
    image_first = to_pil_image(images[0, 0].cpu() * 0.5 + 0.5)
    image_first.save(image_path)
    print(f"Saved input image to {image_path}")

    # Vision features
    B, T, C, H, W = images.shape
    images_reshaped = images.view(B * T, C, H, W)
    with torch.no_grad():
        img_feats = vision_encoder(images_reshaped)  # [B*T, F]
        img_feats = img_feats.view(B, T, -1)          # [B, T, F]

    # Global conditioning
    imuV = imuV.unsqueeze(-1)
    imuOmg = imuOmg.unsqueeze(-1)
    posX = posX.unsqueeze(-1)
    posY = posY.unsqueeze(-1)
    global_cond = torch.cat([img_feats, imuV, imuOmg, posX, posY], dim=-1).flatten(start_dim=1)  # [B, T*(F+4)]

    # Initialize noisy velocity
    timestep = torch.full((1,), diffusion_scheduler.config.num_train_timesteps - 1, dtype=torch.long, device=device)
    init_noise = torch.randn_like(batch["actions"][idx].unsqueeze(0)).to(device)
    noisy_velocities = diffusion_scheduler.add_noise(batch["actions"][idx].unsqueeze(0).to(device), init_noise, timestep)
    denoised_velocities = noisy_velocities.clone()

    # Plot setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_title("Linear Velocity (V) over Sequence")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("V (m/s)")
    ax1.set_xlim(0, true_velocities.shape[0])
    ax1.set_ylim(-2.5, 2.5)

    ax2.set_title("Angular Velocity (ω) over Sequence")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("ω (rad/s)")
    ax2.set_xlim(0, true_velocities.shape[0])
    ax2.set_ylim(-2.5, 2.5)

    timesteps = np.arange(true_velocities.shape[0])
    ax1.plot(timesteps, true_velocities[:, 0], "g-", label="True V")
    ax2.plot(timesteps, true_velocities[:, 1], "g-", label="True ω")
    noisy_v_plot, = ax1.plot([], [], "r--", label="Noisy V")
    denoised_v_plot, = ax1.plot([], [], "b-", label="Denoised V")
    noisy_omg_plot, = ax2.plot([], [], "r--", label="Noisy ω")
    denoised_omg_plot, = ax2.plot([], [], "b-", label="Denoised ω")

    ax1.legend()
    ax2.legend()

    def update(step):
        nonlocal denoised_velocities
        t = torch.full((1,), diffusion_scheduler.config.num_train_timesteps - step - 1, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = noise_pred_net(denoised_velocities, t, None, global_cond)
            denoised_velocities = diffusion_scheduler.step(pred, t, denoised_velocities).prev_sample

        noisy_np = noisy_velocities.view(-1, 2).cpu().numpy()
        denoised_np = denoised_velocities.view(-1, 2).detach().cpu().numpy()

        noisy_v_plot.set_data(timesteps, noisy_np[:, 0])
        denoised_v_plot.set_data(timesteps, denoised_np[:, 0])
        noisy_omg_plot.set_data(timesteps, noisy_np[:, 1])
        denoised_omg_plot.set_data(timesteps, denoised_np[:, 1])

        return denoised_v_plot, noisy_v_plot, denoised_omg_plot, noisy_omg_plot

    writer = FFMpegWriter(fps=5, metadata=dict(title="Denoising Velocities"))
    with writer.saving(fig, video_path, dpi=100):
        for step in tqdm(range(num_steps)):
            update(step)
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved video at {video_path}")



if __name__ == "__main__":
    # Load dataset
    dataset = CustomDataset(
        #csv_file=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\csv_files\TSyn_data_filtered.csv",
        csv_file=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset.csv",
        image_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((96, 96)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
        input_seq=10, output_seq=16
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # Load models
    vision_encoder, noise_pred_net = load_model(r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\trained_policyV1_pos.pth")

    # Run validation
    validate(dataloader, vision_encoder, noise_pred_net,
             num_steps=100,
             video_path=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\videos\outputVideos\denoising_trajectory.mp4",
             image_path=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\videos\outputVideos\input_image.png")
