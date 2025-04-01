import cv2

def load_frames_from_mp4(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Failed to open video file: {filename}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
