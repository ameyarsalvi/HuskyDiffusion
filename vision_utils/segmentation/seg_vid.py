import cv2
import numpy as np

# Set the RGB bounds for brown shades
lower_bound = np.array([97, 85, 12], dtype=np.uint8)  # Example values
upper_bound = np.array([255, 180, 115], dtype=np.uint8)

# Open the video file
video_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\animation_bag_9.mp4"  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Unable to open video file: {video_path}")

# Get the video's frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Delay between frames in milliseconds

# Create a window
cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Masked Video", cv2.WINDOW_NORMAL)

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a mask for the brown shades
    mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

    # Apply the mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original and masked frames
    cv2.imshow("Original Video", frame)
    cv2.imshow("Masked Video", masked_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
