import time
from models.unsupervised_tracking import object_tracking
from preprocessing import video_loader
from utils.visualization import Visualization

video_rgb = video_loader.load_frames_from_mp4('../data/man_rgb.mp4')
video_depth = video_loader.load_frames_from_mp4('../data/man_depth.mp4')

start_time = time.time()
# Visualization.visualize_spatio_color_granules(video_rgb[21], "../results/man/color_2.jpg", 10, False)
# Visualization.visualize_spatiotemporal_granules(video_rgb[18:22], "../results/man/sp_t_2.jpg", 40, True)
# Visualization.visualize_rgb_granules(video_rgb[18:22], "../results/man/rgb_2.jpg", 40, True)
# Visualization.visualize_d_granules(video_depth[18:22], "../results/man/depth_1.jpg", True)

#
object_tracking(video_rgb, video_depth, "../results/man/tracked_man_full_t10_p4.mp4", 10, 4)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Czas wykonania: {elapsed_time:.5f} sekund")
