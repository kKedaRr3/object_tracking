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


video = video_loader.load_frames_from_mp4('../data/spoon.mp4')[:20]
print(video[0].shape)
difference = compute_3D_difference_matrix(video)

Visualization.visualize_video_granulation(difference, "../results/difference_granulation_spoon_th2.avi", 2)
# Visualization.visualize_image_granulation(difference[5], "../results/diff_spoon_th200.jpg", 200)
