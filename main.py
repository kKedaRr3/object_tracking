import cv2
from models.rulebase import generate_rule_base
from preprocessing import video_loader
from preprocessing.granulation import *
from utils.visualization import Visualization

video = video_loader.load_frames_from_mp4('../data/spoon.mp4')[10:]

# Visualization.visualize_spatio_color_granules(diff[0], "../results/spoon/granules_rgb_diff_with_bbox_th67.jpg", 67, True, True, (255, 0, 0))
# Visualization.visualize_spatiotemporal_granules(video[:5], "../results/spoon/spatiotemporal_granules_with_bbox_3.jpg", 3, True)
# Visualization.visualize_rgb_d_granules(video[:5], "../results/spoon/rgb_d_granules.jpg", 3)

def test_rule_base(video):
    threshold = 67
    p = 3
    frames = video

    initial_frames = frames[:p]
    subsequent_frame = frames[p]

    diff_3D_matrix = compute_3D_difference_matrix(initial_frames, subsequent_frame)
    median_matrix = compute_median_matrix(diff_3D_matrix)
    med_threshold = 0.3 * np.max(median_matrix)

    # _gib bo ktotka z granules(g), initial_colors(i) i bounding_boxes(b)
    spatio_temporal_gib = form_spatiotemporal_granules(diff_3D_matrix, med_threshold)

    rgb_gib = form_rgb_d_granules(spatio_temporal_gib[0], spatio_temporal_gib[1], spatio_temporal_gib[2], threshold)

    # tylko do testow
    # d_gib = create_granules_color(median_matrix, threshold)
    d_gib = rgb_gib

    subsequent_spatio_colour_gib = create_granules_color(subsequent_frame, threshold)

    rule_base, features = generate_rule_base(subsequent_spatio_colour_gib, spatio_temporal_gib, rgb_gib, d_gib)

    rule_base_scaled = (rule_base / 2 * 255).astype(np.uint8)
    cv2.imwrite("../results/spoon/rule_base_3.jpg", rule_base_scaled)




# cos sie liczy i jakis wynik wychodzi ale czy jest dobry to niewiadomo (trzeba sprawdzic jeszcze na poprawnych granulach glebokosciwych)
test_rule_base(video)


