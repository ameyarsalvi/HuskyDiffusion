import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_cmd_heatmap(csv_path, start_row, end_row, resolution=0.1):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Subset the rows
    df = df.iloc[start_row:end_row].copy()
    
    # Binning cmd_v and cmd_omg
    v_bins = np.arange(0, 1 + resolution, resolution)
    omg_bins = np.arange(-1, 1 + resolution, resolution)
    
    heatmap = np.zeros((len(omg_bins) - 1, len(v_bins) - 1))

    # Fill heatmap
    for v, omg in zip(df["cmd_v"], df["cmd_omg"]):
        v_idx = np.digitize(v, v_bins) - 1
        omg_idx = np.digitize(omg, omg_bins) - 1
        if 0 <= v_idx < heatmap.shape[1] and 0 <= omg_idx < heatmap.shape[0]:
            heatmap[omg_idx, v_idx] += 1

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    extent = [v_bins[0], v_bins[-1], omg_bins[0], omg_bins[-1]]
    plt.imshow(heatmap, origin='lower', extent=extent, aspect='auto', cmap='hot')
    plt.colorbar(label="Frequency")
    plt.xlabel("cmd_v")
    plt.ylabel("cmd_omg")
    plt.title(f"cmd_v vs cmd_omg Heatmap (rows {start_row} to {end_row})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Example usage:
plot_cmd_heatmap(r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_sim\modular_data.csv", start_row=2980, end_row=2980+8)

#2980