'''
Diffusion based training script based on the diffusion package put together by TRI (modules recreated)
by me
'''

### Libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import cv2
import pandas as pd
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
import numpy as np
import math
import os
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/diffusion_training")

#Import all the modules here
import sys
sys.path.insert(0,"C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/")


### Import Diffusion Modules
from modules.resnet import get_resnet50
from modules.resnet import get_resnet18
from modules.datasetv4 import CustomDataset
from modules.unet2 import ConditionalUnet1D  # Import Conv1D module

'''
DDPMScheduler : Provides the relation between nth timestep and the amount of noise that needs
to be added to the image/action/label at that time step. The noise value thus becomes our "truth"
over which optimization happens
'''
#diffusion_scheduler = DDPMScheduler(num_train_timesteps=100)

diffusion_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2"
)


SAVE_DIR = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\checkpointsV1"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_checkpoint(epoch, model, ema_model, optimizer, save_dir=SAVE_DIR):
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

save_policy_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\policies"
os.makedirs(save_policy_path, exist_ok=True)
def save_final_model(model, ema_model, vision_encoder, save_path= os.path.join(save_policy_path, "cone_rig_norm.pth")):
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'vision_encoder_state_dict': vision_encoder.state_dict(),
    }, save_path)
    print(f"Final model saved: {save_path}")



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


### Training Loop
def train():
    '''
    > Read the CSV file (contains image paths and state measurements )
    > Load data for following process (Training/inference)
    
    Training Condition : |Img - V - Omg|Img - V - Omg|Img - V - Omg|...|Img - V - Omg| (x25)
    Policy Output : |X Y|X Y|X Y|...|X Y| (x100)

    '''
    dataset = CustomDataset(
        #csv_file=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\csv_files\TSyn_data_filtered.csv",
        csv_file=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_drive_rig\mod_output_cone_rig.csv",
        base_dir = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_drive_rig",
        image_transform=transforms.Compose([
            #transforms.Lambda(lambda img: transforms.functional.crop(img, top=288, left=0, height=192, width=640)),
            transforms.Resize((96, 96)),
            transforms.Lambda(hsv_threshold_pil),
            #transforms.Resize((160, 48)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
        input_seq=2, output_seq=8
    )

    
    #Visualize some images from the dataset
    print("Visualizing some images from the dataset:")
    for i in range(2):  # Choose a few samples
        sample = dataset[i]
        img_seq = sample['images']  # Shape: [input_seq, C, H, W]

        for t in range(img_seq.shape[0]):  # Loop over sequence
            img_tensor = img_seq[t]  # Shape: [C, H, W]
            img = img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5  # De-normalize
            img = np.clip(img, 0, 1)  # Just in case values exceed bounds
            plt.imshow(img)
            plt.title(f"Sample {i}, Timestep {t}")
            plt.axis('off')
            plt.show()
    

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    vision_encoder = get_resnet18().to(device)
    #for param in vision_encoder.parameters():
    #    param.requires_grad = False


    '''
        input_dim= Input dimension for actions (PosX and PosY),
        local_cond_dim= Size of local condtioning (default 16),
        global_cond_dim= Size of global condition ([Images, V, Omega] x 25, flattened),
        diffusion_step_embed_dim= Timestep > SineEmbed > NN to amplify effect,
        down_dims=[256, 512, 1024] : Downmoudules, standard,
        kernel_size= KS for Conv1D blocks in ConvResNet,
        n_groups=groups for Conv1D blocks in ConvResNet
    '''
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        local_cond_dim=16,
        #global_cond_dim=12900, #(if Resnet50 : 2048*25 + 1*25 + 1*25 = 51250) (if Resnet18 : 512*25 + 1*25 + 1*25 + 1*25 + 1*25 = 12850)
        #global_cond_dim=2580, #(if Resnet50 : 2048*25 + 1*25 + 1*25 = 51250) (if Resnet18 : 512*25 + 1*25 + 1*25 + 1*25 + 1*25 = 12850)
        global_cond_dim=2*(512+1+1+1+1),
        diffusion_step_embed_dim=256,
        #down_dims=[256, 512, 1024],
        down_dims=[128, 256, 512],
        kernel_size=3,
        n_groups=8).to(device)

    # A decay parameter to smoothly update model weights
    ema = EMAModel(parameters=noise_pred_net.parameters())

    '''
    ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75,           # controls warmup behavior (optional)
    decay=0.9          # default 
    )
    '''

    num_epochs = 100
    #optimizer = torch.optim.AdamW(noise_pred_net.parameters(), lr=1e-5, weight_decay=1e-2)
    params = list(noise_pred_net.parameters()) + list(vision_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


    for epoch in range(num_epochs):
        epoch_loss = []
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            ### Sample data from dataloader (batchsize x data_dims)
            images = batch["images"].to(device)    # (64, 25, 3, 96, 96)
            imu_v = batch['imu_v'].to(device).float()      # (64, 25)
            imu_omg = batch['imu_omg'].to(device).float()  # (64, 25)
            wheel_L = batch['wheel_L'].to(device).float()  # (64, 25)
            wheel_R = batch['wheel_R'].to(device).float()  # (64, 25)
            actions = batch["actions"].to(device).float() # (64, 100, 2)

            ### Encode Image features 
            B, T, C, H, W = images.shape  # [64, 25, 3, 96, 96]
            images = images.view(B * T, C, H, W).to(device)   # -> [1600, 3, 96, 96]
            image_features = vision_encoder(images)
            image_features = image_features.view(B, T, -1)


            imu_v = imu_v.unsqueeze(-1)      # [B, T, 1]
            imu_omg = imu_omg.unsqueeze(-1)  # [B, T, 1]
            wheel_L = wheel_L.unsqueeze(-1)      # [B, T, 1]
            wheel_R = wheel_R.unsqueeze(-1)  # [B, T, 1]

            global_cond = torch.cat([image_features, imu_v, imu_omg, wheel_L, wheel_R], dim=-1)  # [B, T, F+4]
            global_cond = global_cond.flatten(start_dim=1).to(dtype=torch.float32, device = device)  # [B, T*(F+4)]

            local_cond = None

            # Sample timesteps
            timestep = torch.randint(0, diffusion_scheduler.config.num_train_timesteps, (actions.size(0),), device=device)

            # Generate noise
            noise = torch.randn_like(actions, device=device)
            # Add noise using DDPM method
            noisy_actions = diffusion_scheduler.add_noise(actions, noise, timestep)

            # Predict noise
            pred_noise = noise_pred_net(noisy_actions, timestep, local_cond, global_cond)
            
            # Compute loss
            noise_loss = nn.MSELoss()(pred_noise.squeeze(1), noise)  # ✅ Ensure same shape

            #with torch.no_grad():
            #    denoised_actions = diffusion_scheduler.step(pred_noise, timestep, noisy_actions).prev_sample

            denoised_actions = []
            for i in range(noisy_actions.shape[0]):
                result = diffusion_scheduler.step(
                    pred_noise[i].unsqueeze(0), 
                    timestep[i].item(), 
                    noisy_actions[i].unsqueeze(0)
                )
                denoised_actions.append(result.prev_sample)

            denoised_actions = torch.cat(denoised_actions, dim=0)

            lmb = 0.0
            trajectory_loss = nn.MSELoss()(denoised_actions, actions)
            
            total_loss = noise_loss + lmb * trajectory_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(noise_pred_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update EMA
            ema.step(noise_pred_net.parameters())  # ✅ Correct EMA update

            # Store loss
            epoch_loss.append(total_loss.item())

            # ✅ TensorBoard Logging
            writer.add_scalar("Loss/train", total_loss.item(), epoch * len(dataloader) + step)

        # Scheduler step
        lr_scheduler.step()

        # Save checkpoint
        #save_checkpoint(epoch, noise_pred_net, ema, optimizer)


    # Close TensorBoard writer after training completes
    writer.close()
    save_final_model(noise_pred_net, ema, vision_encoder)



if __name__ == "__main__":
    train()