import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import UnivariateSpline


def animate_prediction(index, prediction_dir, save_video=False, save_final_image = False):
    assert 1 <= index <= 7828, "Index must be between 1 and 100"

    # File paths
    file_name = f"pred_seq_{index-1:03d}.csv"
    file_path = os.path.join(prediction_dir, file_name)

    # Load data
    df = pd.read_csv(file_path)
    cmd_V_true = df["cmd_V_true"].values
    cmd_Omg_true = df["cmd_Omg_true"].values

    # Stack predictions [16 x 100]
    pred_V = np.stack([df[f"pred_V_step_{k+1}"].values for k in range(100)], axis=1)
    pred_Omg = np.stack([df[f"pred_Omg_step_{k+1}"].values for k in range(100)], axis=1)

    # Time axis
    t = np.arange(len(cmd_V_true))
    t_fine = np.linspace(0, len(t)-1, 10 * len(t))  # 10x denser query points

    # Setup plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # V plot
    ax1.plot(t, cmd_V_true, "g-", label="True V")
    pred_line_v, = ax1.plot([], [], "b--", label="Pred V (step k)")
    spline_line_v, = ax1.plot([], [], "r-", label="Spline Fit")
    ax1.set_xlim(0, len(t)-1)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title("Linear Velocity (V)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("V (m/s)")
    ax1.grid(True)
    ax1.legend()

    # Omega plot
    ax2.plot(t, cmd_Omg_true, "g-", label="True ω")
    pred_line_omg, = ax2.plot([], [], "b--", label="Pred ω (step k)")
    spline_line_omg, = ax2.plot([], [], "r-", label="Spline Fit")
    ax2.set_xlim(0, len(t)-1)
    ax2.set_ylim(-1, 1)
    ax2.set_title("Angular Velocity (ω)")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("ω (rad/s)")
    ax2.grid(True)
    ax2.legend()

    def update(frame):
        # Get prediction for current step
        pred_v_step = pred_V[:, frame]
        pred_omg_step = pred_Omg[:, frame]

        # Update raw prediction lines
        pred_line_v.set_data(t, pred_v_step)
        pred_line_omg.set_data(t, pred_omg_step)

        # Fit and evaluate spline
        spline_v = UnivariateSpline(t, pred_v_step, s=1)
        spline_omg = UnivariateSpline(t, pred_omg_step, s=1)
        spline_line_v.set_data(t_fine, spline_v(t_fine))
        spline_line_omg.set_data(t_fine, spline_omg(t_fine))

        return pred_line_v, spline_line_v, pred_line_omg, spline_line_omg

    ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

    if save_video:
        from matplotlib.animation import FFMpegWriter
        video_path = os.path.join(prediction_dir, f"prediction_anim_{index:03d}.mp4")
        ani.save(video_path, writer=FFMpegWriter(fps=10))
        print(f"Saved animation to {video_path}")
    else:
        plt.tight_layout()
        plt.show()

    if save_final_image:
                # Save final frame as image
        final_v = pred_V[:, -1]
        final_omg = pred_Omg[:, -1]
        spline_v = UnivariateSpline(t, final_v, s=1)
        spline_omg = UnivariateSpline(t, final_omg, s=1)

        fig_final, (ax1_final, ax2_final) = plt.subplots(2, 1, figsize=(12, 8))

        # Final V
        ax1_final.plot(t, cmd_V_true, "g-", label="True V")
        ax1_final.plot(t, final_v, "b--", label="Pred V (step 100)")
        ax1_final.plot(t_fine, spline_v(t_fine), "r-", label="Spline Fit")
        ax1_final.set_title("Final Linear Velocity (V) - Step 100")
        ax1_final.set_xlabel("Timestep")
        ax1_final.set_ylabel("V (m/s)")
        ax1_final.set_xlim(0, len(t) - 1)
        ax1_final.set_ylim(-0.5, 1.5)
        ax1_final.grid(True)
        ax1_final.legend()

        # Final Omega
        ax2_final.plot(t, cmd_Omg_true, "g-", label="True ω")
        ax2_final.plot(t, final_omg, "b--", label="Pred ω (step 100)")
        ax2_final.plot(t_fine, spline_omg(t_fine), "r-", label="Spline Fit")
        ax2_final.set_title("Final Angular Velocity (ω) - Step 100")
        ax2_final.set_xlabel("Timestep")
        ax2_final.set_ylabel("ω (rad/s)")
        ax2_final.set_xlim(0, len(t) - 1)
        ax2_final.set_ylim(-1, 1)
        ax2_final.grid(True)
        ax2_final.legend()

        plt.tight_layout()
        image_path = os.path.join(prediction_dir, f"prediction_final_{index:03d}.png")
        plt.savefig(image_path, dpi=200)
        plt.close()
        print(f"Saved final frame image to {image_path}")




prediction_dir = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\predictions"
animate_prediction(index=7440, prediction_dir=prediction_dir, save_video=True,save_final_image=True)

#index = 1569, 3832, 6450, 158, 2607, 7440
