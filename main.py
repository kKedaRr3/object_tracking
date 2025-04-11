import cv2
from preprocessing import video_loader
from preprocessing.video_preprocessing import compute_3D_difference_matrix
from utils.visualization import Visualization

video = video_loader.load_frames_from_mp4('../data/spoon.mp4')
print(video[0].shape)
# Visualization.visualize_video_granulation(difference, "../results/granulation_difference_spoon_test_3.avi", 350)


diff = compute_3D_difference_matrix(video[:4], video[4])

# Visualization.visualize_spatio_color_granules(diff[0], "../results/spoon/granules_rgb_diff_with_bbox_th67.jpg", 67, True, True, (255, 0, 0))
# Visualization.visualize_spatiotemporal_granules(video[:5], "../results/spoon/spatiotemporal_granules_with_bbox_3.jpg", 3, True)
Visualization.visualize_rgb_d_granules(video[:5], "../results/spoon/rgb_d_granules.jpg", 3)