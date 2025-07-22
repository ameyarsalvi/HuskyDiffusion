'''
Validation Code for action : Position (X-Y)

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
from modules.datasetv2 import CustomDataset
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
        #global_cond_dim=12900, #(if Resnet50 : 2048*25 + 1*25 + 1*25 = 51250) (if Resnet18 : 512*25 + 1*25 + 1*25 = 12850)
        global_cond_dim=2*(512+1+1+1+1),
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
    
    #diffusion_scheduler = DDPMScheduler(num_train_timesteps=100)
    diffusion_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2"
)
    batch = next(iter(dataloader))
    idx = np.random.randint(0, batch["images"].shape[0])

    # Get sample
    images = batch["images"][idx].unsqueeze(0).to(device)  # [1, 25, 3, 96, 96]
    imuV = batch["imu_v"][idx].unsqueeze(0).to(device).float()  # [1, 25]
    imuOmg = batch["imu_omg"][idx].unsqueeze(0).to(device).float()  # [1, 25]
    posX = batch["posX"][idx].unsqueeze(0).to(device).float()  # [1, 25]
    posY = batch["posY"][idx].unsqueeze(0).to(device).float()  # [1, 25]
    true_actions = batch["actions"][idx].cpu().numpy()  # [100, 2]

    # Save input image (first frame)
    image_first = to_pil_image(images[0, 0].cpu() * 0.5 + 0.5)
    image_first.save(image_path)
    print(f"Saved input image to {image_path}")

    # Vision features
    B, T, C, H, W = images.shape
    images_reshaped = images.view(B * T, C, H, W)
    with torch.no_grad():
        img_feats = vision_encoder(images_reshaped)  # [B*T, F]
        img_feats = img_feats.view(B, T, -1)         # [B, T, F]

    imuV = imuV.unsqueeze(-1)
    imuOmg = imuOmg.unsqueeze(-1)
    posX = posX.unsqueeze(-1)
    posY = posY.unsqueeze(-1)
    global_cond = torch.cat([img_feats, imuV, imuOmg, posX, posY], dim=-1).flatten(start_dim=1)  # [B, T*(F+4)]

    # Initialize noisy trajectory
    timestep = torch.full((1,), diffusion_scheduler.num_train_timesteps - 1, dtype=torch.long, device=device)
    init_noise = 5*torch.randn_like(batch["actions"][idx].unsqueeze(0)).to(device)  # large initial noise
    noisy_actions = diffusion_scheduler.add_noise(batch["actions"][idx].unsqueeze(0).to(device), init_noise, timestep)
    denoised_actions = noisy_actions.clone()

    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("Denoising Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    true_plot, = ax.plot(true_actions[:, 0], true_actions[:, 1], "go", label="True", linestyle="none")
    noisy_plot, = ax.plot([], [], "ro", label="Noisy", linestyle="none")
    denoised_plot, = ax.plot([], [], "bo", label="Denoised", linestyle="none")
    ax.legend()

    def update(step):
        nonlocal denoised_actions
        t = torch.full((1,), diffusion_scheduler.num_train_timesteps - step - 1, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = noise_pred_net(denoised_actions, t, None, global_cond)
            denoised_actions = diffusion_scheduler.step(pred, t, denoised_actions).prev_sample

        noisy_np = noisy_actions.view(-1, 2).cpu().numpy()
        denoised_np = denoised_actions.view(-1, 2).detach().cpu().numpy()

        noisy_plot.set_data(noisy_np[:, 0], noisy_np[:, 1])
        denoised_plot.set_data(denoised_np[:, 0], denoised_np[:, 1])
        return denoised_plot, noisy_plot

    writer = FFMpegWriter(fps=5, metadata=dict(title="Denoising Trajectory"))
    with writer.saving(fig, video_path, dpi=100):
        for step in tqdm(range(num_steps)):
            update(step)
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved video at {video_path}")
    

        # Final plot (static image of true vs denoised trajectory)
    final_denoised_np = denoised_actions.view(-1, 2).detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.plot(true_actions[:, 0], true_actions[:, 1], 'go-', label="True Trajectory")
    plt.plot(final_denoised_np[:, 0], final_denoised_np[:, 1], 'bo-', label="Denoised Trajectory")
    plt.title("Final Denoised vs True Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid(True)
    plt.legend()
    final_plot_path = os.path.splitext(video_path)[0] + "_final.jpg"
    plt.savefig(final_plot_path, dpi=200)
    plt.close()
    print(f"Saved final trajectory plot at {final_plot_path}")



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
        input_seq=2, output_seq=16
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Load models
    vision_encoder, noise_pred_net = load_model(r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\training\velDiff\policy_act_norm.pth")

    # Run validation
    validate(dataloader, vision_encoder, noise_pred_net,
             num_steps=100,
             video_path=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\videos\outputVideos\denoising_trajectory.mp4",
             image_path=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\videos\outputVideos\input_image.png")
