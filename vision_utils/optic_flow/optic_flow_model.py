import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

def compute_robot_motion(wheel_left, wheel_right, r=10, L=20):
    """
    Compute linear and angular velocities of a differential drive robot.
    """
    if wheel_left == 0 and wheel_right == 0:
        return 0, 0  # No motion
    A = np.array([
        [r / 2, r / 2],    # Linear velocity row
        [-r / L, r / L]    # Angular velocity row
    ])
    return A @ np.array([wheel_left, wheel_right])

def compute_planar_depth(image_height, h=40, f_y=687.4716, cy=181.3353):
    """
    Compute depth map under planar assumption.
    """
    v_coords = np.arange(image_height)
    depth_map = h * f_y / (v_coords - cy)
    depth_map[v_coords <= cy] = np.nan  # Ignore values above the horizon
    depth_map = np.nan_to_num(depth_map, nan=1e6)  # Replace NaN with a large value
    return depth_map

def compute_optical_flow_planar(V, omega, image_shape, h=40, f_x=690.1305, f_y=687.4716, cx=258.1100, cy=181.3353):
    """
    Compute dense optical flow for a planar scene given robot motion and camera parameters.
    """
    image_height, image_width = image_shape
    u, v = np.meshgrid(np.arange(image_width), np.arange(image_height))

    # Compute planar depth map
    depth_map = compute_planar_depth(image_height, h=h, f_y=f_y, cy=cy)
    depth_map = depth_map[:, np.newaxis]  # Expand to 2D for broadcasting

    # Convert image coordinates to camera frame
    X = (u - cx) * depth_map / f_x
    Y = (v - cy) * depth_map / f_y
    Z = depth_map

    # Compute optical flow
    flow_u = (f_x / Z) * (V - omega * Y)
    flow_v = (f_y / Z) * (omega * X)

    return flow_u, flow_v

def visualize_and_save_optical_flow(csv_file, output_file, r=16.5, L=55.5, dt=0.1, image_shape=(480, 640), h=40, 
                                    f_x=690.1305, f_y=687.4716, cx=258.1100, cy=181.3353):
    """
    Visualize and save optical flow animation based on wheel velocities from a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file containing wheel velocities.
        output_file (str): Path to save the video file (e.g., "output.mp4").
        r (float): Radius of the wheels (cm).
        L (float): Distance between the wheels (cm).
        dt (float): Time interval between measurements (s).
        image_shape (tuple): Shape of the image (height, width).
        h (float): Camera height above the ground (cm).
        f_x (float): Focal length along the x-axis (pixels).
        f_y (float): Focal length along the y-axis (pixels).
        cx, cy (float): Principal point of the camera (pixels).
    """
    # Read wheel velocities from the CSV file
    data = pd.read_csv(csv_file, header=None).values
    left_wheel_vals = data[0]  # First row: left wheel velocities
    right_wheel_vals = data[1]  # Second row: right wheel velocities

    assert len(left_wheel_vals) == len(right_wheel_vals), "Wheel velocity lists must have the same length!"

    # Create a down-sampled grid for visualization
    step = 10
    x, y = np.meshgrid(np.arange(0, image_shape[1], step), np.arange(0, image_shape[0], step))

    # Create figure for animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Optical Flow Vectors")
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)  # Flip y-axis for correct orientation
    quiver = ax.quiver(x, y, np.zeros_like(x), np.zeros_like(y), scale=50, pivot='middle')

    def update(frame_idx):
        # Compute robot motion
        V, omega = compute_robot_motion(left_wheel_vals[frame_idx], right_wheel_vals[frame_idx], r=r, L=L)

        # Compute optical flow
        flow_u, flow_v = compute_optical_flow_planar(V, omega, image_shape, h=h, f_x=f_x, f_y=f_y, cx=cx, cy=cy)

        # Down-sample for visualization
        flow_u_downsampled = flow_u[::step, ::step]
        flow_v_downsampled = flow_v[::step, ::step]

        # Update quiver plot
        quiver.set_UVC(flow_u_downsampled.ravel(), flow_v_downsampled.ravel())
        return quiver,

    ani = animation.FuncAnimation(fig, update, frames=len(left_wheel_vals), interval=dt * 1000, blit=True)

    # Save the animation as a video
    print(f"Saving video to {output_file}...")
    ani.save(output_file, writer="ffmpeg", fps=1 / dt)
    print("Video saved successfully!")

    plt.show()

# Example usage
csv_file_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\joint_states.csv"  # Replace with your CSV file path
output_video_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\optic_flow_video.mp4"  # Replace with desired output video path
visualize_and_save_optical_flow(csv_file_path, output_video_path)
