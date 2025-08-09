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

#Import all the modules here
import sys
sys.path.insert(0,"C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/")

# Import modules
from modules.resnet import get_resnet50
from modules.resnet import get_resnet18
from modules.datasetv4 import CustomDataset
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
import csv

def validate_sequential(dataloader, vision_encoder, noise_pred_net, 
                        num_steps=100, output_dir="predictions/", map_file="image_to_csv_mapping.csv"):
    
    os.makedirs(output_dir, exist_ok=True)
    mapping_path = os.path.join(output_dir, map_file)
    mapping_entries = []

    # Diffusion scheduler
    diffusion_scheduler = DDPMScheduler(
        num_train_timesteps=num_steps,
        beta_schedule="squaredcos_cap_v2"
    )

    # Loop over first 100 sequential entries in dataloader
    for i, batch in enumerate(dataloader):
        if i >= 7845:
            break

        # Get batch index 0 from this single-item batch
        images = batch["images"][0].unsqueeze(0).to(device)         # [1, T, C, H, W]
        imu_v = batch["imu_v"][0].unsqueeze(0).to(device).float()   # [1, T]
        imu_omg = batch["imu_omg"][0].unsqueeze(0).to(device).float()
        wheel_L = batch["wheel_L"][0].unsqueeze(0).to(device).float()
        wheel_R = batch["wheel_R"][0].unsqueeze(0).to(device).float()
        true_actions = batch["actions"][0].cpu().numpy()            # [16, 2]

        # Prepare vision features
        B, T, C, H, W = images.shape
        with torch.no_grad():
            img_feats = vision_encoder(images.view(B*T, C, H, W))  # [B*T, F]
            img_feats = img_feats.view(B, T, -1)                   # [1, T, F]

        # Create conditioning input
        imu_v = imu_v.unsqueeze(-1)
        imu_omg = imu_omg.unsqueeze(-1)
        wheel_L = wheel_L.unsqueeze(-1)
        wheel_R = wheel_R.unsqueeze(-1)
        global_cond = torch.cat([img_feats, imu_v, imu_omg, wheel_L, wheel_R], dim=-1).flatten(start_dim=1)

        # Init noisy sample
        timestep = torch.full((1,), diffusion_scheduler.num_train_timesteps - 1, dtype=torch.long, device=device)
        init_noise = torch.randn_like(batch["actions"][0].unsqueeze(0)).to(device)
        denoised = diffusion_scheduler.add_noise(batch["actions"][0].unsqueeze(0).to(device), init_noise, timestep)

        # To store predictions at each step
        pred_V = []
        pred_Omg = []

        for step in range(num_steps):
            t = torch.full((1,), diffusion_scheduler.num_train_timesteps - step - 1, dtype=torch.long, device=device)
            with torch.no_grad():
                noise_pred = noise_pred_net(denoised, t, None, global_cond)
                denoised = diffusion_scheduler.step(noise_pred, t, denoised).prev_sample

            den_np = denoised.view(-1, 2).detach().cpu().numpy()
            pred_V.append(den_np[:, 0])     # shape: [16]
            pred_Omg.append(den_np[:, 1])   # shape: [16]

        # Stack predictions as: [cmd_V_true, predV_step1,...100, cmd_Omg_true, predOmg_step1,...100]
        pred_V_array = np.stack(pred_V, axis=1)          # [16, 100]
        pred_Omg_array = np.stack(pred_Omg, axis=1)      # [16, 100]
        cmd_V_true = true_actions[:, 0].reshape(-1, 1)    # [16, 1]
        cmd_Omg_true = true_actions[:, 1].reshape(-1, 1)  # [16, 1]

        full_array = np.hstack([cmd_V_true, pred_V_array, cmd_Omg_true, pred_Omg_array])  # [16, 202]

        # Save CSV
        pred_filename = f"pred_seq_{i:03d}.csv"
        pred_filepath = os.path.join(output_dir, pred_filename)
        header = ["cmd_V_true"] + [f"pred_V_step_{j+1}" for j in range(100)] + \
                 ["cmd_Omg_true"] + [f"pred_Omg_step_{j+1}" for j in range(100)]
        np.savetxt(pred_filepath, full_array, delimiter=",", header=",".join(header), comments='')

        # Log image-to-file mapping
        image_name = batch["images"][0][0].cpu()  # first image in the input sequence
        image_file = dataloader.dataset.data.loc[i, 'image']  # assumes 'image' column has filename
        mapping_entries.append([image_file, pred_filename])

        print(f"[{i+1}/100] Saved: {pred_filename}")

    # Save mapping file
    with open(mapping_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "csv_file"])
        writer.writerows(mapping_entries)

    print(f"\nMapping file saved at {mapping_path}")





if __name__ == "__main__":
    # Load dataset
    dataset = CustomDataset(
        csv_file=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_drive_rig\mod_output_cone_rig.csv",
        base_dir = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_drive_rig",
        image_transform=torchvision.transforms.Compose([
            #torchvision.transforms.Lambda(lambda img: torchvision.transforms.functional.crop(img, top=288, left=0, height=192, width=640)),
            torchvision.transforms.Resize((96, 96)),
            torchvision.transforms.Lambda(hsv_threshold_pil),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
        input_seq=2, output_seq=16
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load models
    vision_encoder, noise_pred_net = load_model(r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\policies\cone_rig_norm.pth")

    # Run validation
    validate_sequential(dataloader, vision_encoder, noise_pred_net,
                    num_steps=100,
                    output_dir=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\predictions",
                    map_file="image_to_csv_mapping.csv")
