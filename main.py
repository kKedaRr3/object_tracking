import time

import cv2

from models.unsupervised_tracking import object_tracking
from preprocessing import video_loader
from utils.visualization import Visualization

video = video_loader.load_frames_from_mp4('../data/spoon.mp4')[1:]


object_tracking(video, "../results/spoon/tracked_spoon.mp4", 50, 3)

# start_time = time.time()
# Visualization.visualize_spatiotemporal_granules(video[1:5], "../results/spoon/test/2.jpg", 3, False)
# Visualization.visualize_spatio_color_granules(video[5], "../results/spoon/test/color_1.jpg", 50, False)
#
# end_time = time.time()
#
# elapsed_time = end_time - start_time
# print(f"Czas wykonania: {elapsed_time:.5f} sekund")