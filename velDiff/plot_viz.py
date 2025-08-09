import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FFMpegWriter

import os
from PIL import Image

#csv file path
csv_file=r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_drive_rig\modular_cone_rig_data.csv"

data = pd.read_csv(csv_file)

rows = data.iloc[1:500]

cmd_v = (0.165/2)*(rows['wheel_L'] + rows['wheel_L'])
cmd_omg = (0.165/0.555)*(-1*rows['wheel_L'] + rows['wheel_L'])

sequence = list(range(1, 500))


plt.subplot(2,1,1)
plt.plot(sequence,cmd_v)
plt.title('Cmd V')

plt.subplot(2,1,2)
plt.plot(sequence,cmd_omg)
plt.title('Cmd Omega')

plt.show()