'''
Batch Evaluations, without plots, only to collect evaluation data

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
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
from sklearn.metrics import mean_absolute_error
import sys



#============= Parse Arguments ==================

###### Extract var arguments from json ##########
import json

with open("C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/velDiff/conf/validate_config.json","r") as f:
    config = json.load(f)

args = config["args"][0]

#================================================
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Import all the modules here
sys.path.insert(0,args['path_system_path'])
# Import modules
from modules.resnet import get_resnet18
from modules.datasetv5 import CustomDataset
from modules.unet2 import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Load model
def load_model(checkpoint_path=args['policy_name']):
    # Define architecture
    vision_encoder = get_resnet18(fc_layer=args['VE_fcLayer']).to(device)
    noise_pred_net = ConditionalUnet1D(
        input_dim=args['train_config_input_dim'],
        local_cond_dim=args['train_config_local_cond_dim'],
        global_cond_dim = args['input_seq_len']*(args['VE_fcLayer'] + (len(args['conditions'])-1)),
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
    #batch = next(iter(dataloader))
    #idx = np.random.randint(0, batch[args['conditions'][0]].shape[0])

    maeV = []
    maeOmg = []
    sample_no = []

    for sample, batch in enumerate(dataloader):
        print(f"sample is {sample}")
        if sample >= 50:
            break

        # Get sample
        images = batch[args['conditions'][0]][0].unsqueeze(0).to(device)    # (64, 25, 3, 96, 96)


        # Vision features
        B, T, C, H, W = images.shape
        images_reshaped = images.view(B * T, C, H, W)
        with torch.no_grad():
            img_feats = vision_encoder(images_reshaped)  # [B*T, F]
            img_feats = img_feats.view(B, T, -1)         # [B, T, F]

        cond = []
        cond_tensor = []
        for i in range(1,len(args['conditions'])):
            cond = batch[args['conditions'][i]][0].unsqueeze(0).to(device).float()
            cond = cond.unsqueeze(-1)
            cond_tensor.append(cond)

        true_velocities = batch["actions"][0].cpu().numpy()  # [100, 2]
        print(f"shape of true vel is{np.shape(true_velocities)}")
        global_cond = torch.cat([img_feats] + cond_tensor, dim=-1).flatten(start_dim=1)  # [B, T, F+5]


        # Initialize noisy trajectory
        timestep = torch.full((1,), diffusion_scheduler.num_train_timesteps - 1, dtype=torch.long, device=device)
        init_noise = torch.randn_like(batch["actions"][0].unsqueeze(0)).to(device)  # large initial noise
        noisy_velocities = diffusion_scheduler.add_noise(batch["actions"][0].unsqueeze(0).to(device), init_noise, timestep)
        denoised_velocities = noisy_velocities.clone()

        def update(step):
            nonlocal denoised_velocities
            t = torch.full((1,), diffusion_scheduler.config.num_train_timesteps - step - 1,
                        dtype=torch.long, device=device)
            with torch.no_grad():
                pred = noise_pred_net(denoised_velocities, t, None, global_cond)
                denoised_velocities = diffusion_scheduler.step(pred, t, denoised_velocities).prev_sample

            noisy_np = noisy_velocities.view(-1, 2).cpu().numpy()
            denoised_np = denoised_velocities.view(-1, 2).detach().cpu().numpy()

            '''
            # 2D updates (unchanged)
            noisy_v_plot.set_data(timesteps, noisy_np[:, 0])
            denoised_v_plot.set_data(timesteps, denoised_np[:, 0])
            noisy_omg_plot.set_data(timesteps, noisy_np[:, 1])
            denoised_omg_plot.set_data(timesteps, denoised_np[:, 1])
            '''

            return noisy_np, denoised_np
    

        for step in tqdm(range(num_steps)):
            noisy_np, denoised_np = update(step) 
        
        # Extract final denoised velocities as numpy (Final 2-D Path)
        final_denoised_np = denoised_np
        timesteps = np.arange(true_velocities.shape[0])

        mae_v = mean_absolute_error(true_velocities[:,0], final_denoised_np[:,0])
        mae_omg = mean_absolute_error(true_velocities[:,1], final_denoised_np[:,1])

        maeV.append(mae_v)
        maeOmg.append(mae_omg)
        sample_no.append(sample)
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.plot(sample_no,maeV)
    ax1.set_title('MAE V')

    ax2.plot(sample_no,maeOmg)
    ax2.set_title('MAE V')

    plt.show()


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
        input_seq=args['input_seq_len'], output_seq=args['output_seq_len']
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load models
    vision_encoder, noise_pred_net = load_model(args['path_saved_policy']+args['policy_name'])

    # Run validation
    validate(dataloader, vision_encoder, noise_pred_net,
             num_steps=args['train_params_schedule_config_steps'],
             video_path=args['save_vid_img_dir']+args['save_traj_video_as'],
             plot_path = args['save_vid_img_dir']+args['save_traj_plot_as'],
             image_path=args['save_vid_img_dir']+args['save_input_img_as'])
