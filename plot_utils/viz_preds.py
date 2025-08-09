import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# --- CONFIG ---
prediction_dir = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\predictions"
start_idx = 3000  # inclusive (0-based index)
end_idx = 4000    # exclusive

# --- Collect matching files ---
all_files = sorted(glob(os.path.join(prediction_dir, "pred_seq_*.csv")))
selected_files = all_files[start_idx:end_idx]

# --- Initialize containers ---
all_true_V = []
all_pred_V = []
all_true_Omg = []
all_pred_Omg = []

# --- Loop over selected files ---
for file_path in selected_files:
    df = pd.read_csv(file_path)
    all_true_V.append(df["cmd_V_true"].values)
    all_pred_V.append(df["pred_V_step_100"].values)
    all_true_Omg.append(df["cmd_Omg_true"].values)
    all_pred_Omg.append(df["pred_Omg_step_100"].values)

# --- Concatenate all ---
V_true_concat = pd.Series(np.concatenate(all_true_V))
V_pred_concat = pd.Series(np.concatenate(all_pred_V))
Omg_true_concat = pd.Series(np.concatenate(all_true_Omg))
Omg_pred_concat = pd.Series(np.concatenate(all_pred_Omg))

# --- Plotting ---
plt.figure(figsize=(16, 6))

# Linear velocity
plt.subplot(2, 1, 1)
plt.plot(V_true_concat, label="True V", color="green")
plt.plot(V_pred_concat, label="Predicted V (Step 100)", color="blue", linestyle="--")
plt.title(f"cmd_V: True vs Predicted (Step 100)")
plt.xlabel("Time index")
plt.ylabel("Linear Velocity (V)")
plt.grid(True)
plt.legend()

# Angular velocity
plt.subplot(2, 1, 2)
plt.plot(Omg_true_concat, label="True ω", color="green")
plt.plot(Omg_pred_concat, label="Predicted ω (Step 100)", color="blue", linestyle="--")
plt.title(f"cmd_Omega: True vs Predicted (Step 100)")
plt.xlabel("Time index")
plt.ylabel("Angular Velocity (ω)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
