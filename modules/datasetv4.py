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
        self.data = self.raw_data  
        self.image_transform = image_transform
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.base_dir = base_dir

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
        wheel_r = input_rows['wheel_R'].values


        # Output sequence (future trajectory)
        cmd_l = output_rows['wheel_L'].values
        cmd_r = output_rows['wheel_R'].values

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
            np.stack((cmd_l, cmd_r), axis=-1),
            dtype=torch.float32
        )

        return {
            'images': image_sequence,              # (input_seq, C, H, W)
            'imu_v': torch.tensor(imu_v),           # (input_seq,)
            'imu_omg': torch.tensor(imu_omg),           # (input_seq,)
            'wheel_L': torch.tensor(wheel_l),       # (input_seq,)
            'wheel_R': torch.tensor(wheel_r),       # (input_seq,)
            'actions': actions                     # (output_seq, 2)
        }
