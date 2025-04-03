import numpy as np
import video_loader
from utils.visualization import Visualization
import cv2


def compute_3D_difference_matrix(frames):
    difference = []
    for frame in frames[:-1]:
        diff_image = cv2.absdiff(frame, frames[-1])

        diff_image = np.clip(diff_image, 0, 255).astype(np.uint8)

        difference.append(diff_image)

    return np.array(difference)


video = video_loader.load_frames_from_mp4('../data/moving_ball.mp4')[:3]
print(len(video))
difference = compute_3D_difference_matrix(video)

Visualization.visualize_video_granulation(video, "../results/granulation_ball_full_test.avi", 200)
# Visualization.visualize_image_granulation(video[2], "../results/ball_th350_test.jpg", 350)
