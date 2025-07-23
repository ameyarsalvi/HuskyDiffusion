import cv2
import numpy as np

# Open video source (file or webcam)
video_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\animation_bag_9.mp4"  # Replace with video file path, or use 0 for a webcam
cap = cv2.VideoCapture(video_path if video_path else 0)

if not cap.isOpened():
    raise FileNotFoundError("Video file not found or camera not accessible!")

# Parameters for Lucas-Kanade Sparse Optical Flow
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read the video!")

# Convert the first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Initialize good features for sparse optical flow
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes (sparse flow)
sparse_mask = np.zeros_like(old_frame)

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Dense Optical Flow
    # -------------------------
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2  # Direction (Hue)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude (Value)
    dense_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # -------------------------
    # Sparse Optical Flow
    # -------------------------
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        sparse_mask = cv2.line(sparse_mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

    sparse_flow = cv2.add(frame, sparse_mask)

    # -------------------------
    # Display Results
    # -------------------------
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Dense Optical Flow", dense_flow)
    cv2.imshow("Sparse Optical Flow", sparse_flow)

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
