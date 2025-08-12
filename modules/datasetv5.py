import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os 


class CustomDataset(Dataset):
    def __init__(self, csv_file, base_dir, image_transform=None, input_seq=25, output_seq=100):
        super().__init__()
        self.raw_data = pd.read_csv(csv_file)
        self.data = self.raw_data.iloc[::9].reset_index(drop=True)  # Step 1: Subsample every 5th row
        #self.data = self.raw_data  
        self.image_transform = image_transform
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.base_dir = base_dir
        self.throttle = 9

    def __len__(self):
        return len(self.data) - (self.input_seq + self.output_seq)

    def __getitem__(self, idx):
        input_rows = self.data.iloc[idx : idx + self.input_seq]
        output_rows = self.data.iloc[idx + self.input_seq : idx + self.input_seq + self.output_seq]

        # Image & IMU input sequence
        image_paths = input_rows['image'].tolist()
        imu_v = input_rows['IMU_v'].values
        imu_omg = input_rows['IMU_omg'].values
        wheel_l = input_rows['wheel_L'].values
        #print(f"wheel_l: {wheel_l}")
        wheel_r = input_rows['wheel_R'].values
        #print(f"wheel_r: {wheel_r}")


        # Output sequence (future trajectory)
        #Calibrated 'radius:r' instead of true r
        cmd_v = (0.1497/2)*(output_rows['wheel_L'].values + output_rows['wheel_R'].values)
        cmd_v_ref = np.full(self.input_seq, np.mean(cmd_v))
        cmd_v_norm = 2 * ((cmd_v + 0.18) / 1.08) - 1
        #print(f"cmd_v: {cmd_v_norm}")
        #Calibrated 'wheel base :B' instead of true B
        #Sign flig between wheel_L and wheel_R to match flipped IMU
        cmd_omg = (0.1497/1.8036)*(1*output_rows['wheel_L'].values - output_rows['wheel_R'].values)
        cmd_omg_norm = 2 * ((cmd_omg + 0.2) / 0.4) - 1
        #print(f"cmd_omg: {cmd_omg_norm}")

        # Load and transform image sequence
        image_sequence = []
        for path in image_paths:
            full_path = os.path.join(self.base_dir, path)  # Join base_dir + relative path
            img = Image.open(full_path).convert('RGB')
            if self.image_transform:
                img = self.image_transform(img)
            image_sequence.append(img)

        image_sequence = torch.stack(image_sequence, dim=0)  # shape: (T, C, H, W)


        actions = torch.tensor(
            np.stack((cmd_v_norm, cmd_omg_norm), axis=-1),
            dtype=torch.float32
        )

        return {
            'images': image_sequence,              # (input_seq, C, H, W)
            'imu_v': torch.tensor(imu_v),           # (input_seq,)
            'imu_omg': torch.tensor(imu_omg),           # (input_seq,)
            'wheel_L': torch.tensor(wheel_l),       # (input_seq,)
            'wheel_R': torch.tensor(wheel_r),       # (input_seq,)
            'ref_velocity' : torch.tensor(cmd_v_ref),
            'actions': actions                     # (output_seq, 2)
        }
