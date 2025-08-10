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

import cv2
from PIL import Image

#============= Parse Arguments ==================

###### Extract var arguments from json ##########
import json

with open("conf/validate_config.json","r") as f:
    config = json.load(f)

args = config["args"][0]

#================================================
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Import all the modules here
import sys
sys.path.insert(0,args['path_system_path'])

# Import modules
from modules.resnet import get_resnet18
from modules.datasetv5 import CustomDataset
from modules.unet2 import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Load model
def load_model(checkpoint_path=args['policy_name']):
    # Define architecture
    vision_encoder = get_resnet18().to(device)
    noise_pred_net = ConditionalUnet1D(
        input_dim=args['train_config_input_dim'],
        local_cond_dim=args['train_config_local_cond_dim'],
        global_cond_dim= args['train_config_global_cond_dim'], #2*(512+1+1+1+1)
        diffusion_step_embed_dim=args['train_config_diffusion_step_embed_dim'],
        down_dims=args['train_config_down_dims'],
        kernel_size=args['train_config_kernel_size'],
        n_groups=args['train_config_n_groups']).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    noise_pred_net.load_state_dict(checkpoint["model_state_dict"])
    vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
    vision_encoder.eval()
    noise_pred_net.eval()

    return vision_encoder, noise_pred_net


def hsv_threshold_pil(pil_img):
    # Convert PIL to numpy array (RGB)
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Convert to HSV and apply threshold
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Convert binary mask to 3-channel grayscale to maintain consistency
    mask_3ch = cv2.merge([mask, mask, mask])  # shape (H, W, 3)
    
    # Convert back to PIL image
    return Image.fromarray(mask_3ch)

# Validation logic
def validate(dataloader, vision_encoder, noise_pred_net, num_steps,
             video_path,plot_path, image_path):
    
    #diffusion_scheduler = DDPMScheduler(num_train_timesteps=100)
    diffusion_scheduler = DDPMScheduler(
    num_train_timesteps=num_steps,
    beta_schedule=args['train_params_schedule_config_scheduler']
)
    batch = next(iter(dataloader))
    idx = np.random.randint(0, batch["images"].shape[0])

    # Get sample
    images = batch["images"][idx].unsqueeze(0).to(device)  # [1, 25, 3, 96, 96]
    imu_v = batch['imu_v'][idx].unsqueeze(0).to(device).float()      # (64, 25)
    imu_omg = batch['imu_omg'][idx].unsqueeze(0).to(device).float()  # (64, 25)
    wheel_L = batch['wheel_L'][idx].unsqueeze(0).to(device).float()  # (64, 25)
    wheel_R = batch['wheel_R'][idx].unsqueeze(0).to(device).float()  # (64, 25)
    ref_velocity = batch['ref_velocity'][idx].unsqueeze(0).to(device).float()  # (64, 25)
    true_velocities = batch["actions"][idx].cpu().numpy()  # [100, 2]

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

    imu_v = imu_v.unsqueeze(-1)      # [B, T, 1]
    imu_omg = imu_omg.unsqueeze(-1)  # [B, T, 1]
    wheel_L = wheel_L.unsqueeze(-1)      # [B, T, 1]
    wheel_R = wheel_R.unsqueeze(-1)  # [B, T, 1]
    ref_velocity = ref_velocity.unsqueeze(-1)

    #print(f"shape for wheel_L is{wheel_L.size()}")
    #print(f"shape for ref_velocity is{ref_velocity.size()}")
    
    global_cond = torch.cat([img_feats, imu_v, imu_omg, wheel_L, wheel_R, ref_velocity], dim=-1).flatten(start_dim=1)

    # Initialize noisy trajectory
    timestep = torch.full((1,), diffusion_scheduler.num_train_timesteps - 1, dtype=torch.long, device=device)
    init_noise = torch.randn_like(batch["actions"][idx].unsqueeze(0)).to(device)  # large initial noise
    noisy_velocities = diffusion_scheduler.add_noise(batch["actions"][idx].unsqueeze(0).to(device), init_noise, timestep)
    denoised_velocities = noisy_velocities.clone()

    
    # Plot setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_title("Left Wheel Velocity (w_L) rad/s over Sequence")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("ω (rad/s)")
    ax1.set_xlim(0, true_velocities.shape[0])
    ax1.set_ylim(-1, 1)

    ax2.set_title("Right Wheel Velocity (w_R) rad/s over Sequence")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("ω (rad/s)")
    ax2.set_xlim(0, true_velocities.shape[0])
    ax2.set_ylim(-1, 1)

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

    # Extract final denoised velocities as numpy
    final_denoised_np = denoised_velocities.view(-1, 2).detach().cpu().numpy()
    timesteps = np.arange(true_velocities.shape[0])

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Linear velocity subplot
    ax1.plot(timesteps, true_velocities[:, 0], 'g-', label="True V")
    ax1.plot(timesteps, final_denoised_np[:, 0], 'b-', label="Denoised V")
    ax1.set_title("Left Wheel Velocity (w_L) rad/s over Sequence")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("ω (rad/s)")
    ax1.set_xlim(0, true_velocities.shape[0])
    ax1.set_ylim(-1, 1)
    ax1.grid(True)
    ax1.legend()

    # Angular velocity subplot
    ax2.plot(timesteps, true_velocities[:, 1], 'g-', label="True ω")
    ax2.plot(timesteps, final_denoised_np[:, 1], 'b-', label="Denoised ω")
    ax2.set_title("Right Wheel Velocity (w_L) rad/s over Sequence")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("ω (rad/s)")
    ax2.set_xlim(0, true_velocities.shape[0])
    ax2.set_ylim(-1, 1)
    ax2.grid(True)
    ax2.legend()

    # Save and close
    final_plot_path = plot_path
    plt.tight_layout()
    plt.savefig(final_plot_path, dpi=200)
    plt.close()
    print(f"Saved final V/ω comparison plot at {final_plot_path}")




if __name__ == "__main__":
    # Load dataset
    dataset = CustomDataset(
        csv_file=args['path_dataset_csv'],
        base_dir = args['path_dataset'],
        image_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((96, 96)),
            torchvision.transforms.Lambda(hsv_threshold_pil),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
        input_seq=2, output_seq=16
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['train_config_data_batch_size'], shuffle=True)

    # Load models
    vision_encoder, noise_pred_net = load_model(args['path_saved_policy']+args['policy_name'])

    # Run validation
    validate(dataloader, vision_encoder, noise_pred_net,
             num_steps=args['train_params_schedule_config_steps'],
             video_path=args['save_vid_img_dir']+args['save_traj_video_as'],
             plot_path = args['save_vid_img_dir']+args['save_traj_plot_as'],
             image_path=args['save_vid_img_dir']+args['save_input_img_as'])
