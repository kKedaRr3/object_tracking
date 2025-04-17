import cv2

from models.flow_graph import generate_flow_graph, compute_rule_base_coverage
from models.rulebase import generate_rule_base
from models.unsupervised_tracking import object_tracking
from preprocessing import video_loader
from preprocessing.granulation import *
from utils.visualization import Visualization

video = video_loader.load_frames_from_mp4('../data/spoon.mp4')[1:]


# Visualization.visualize_spatio_color_granules(video[4], "../results/spoon/granules_rgb_no_bbox_th200.jpg", 200, True, False, (255, 0, 0))
Visualization.visualize_spatiotemporal_granules(video[:5], "../results/spoon/test/spatiotemporal_granules_with_bbox_3.jpg", 3, True)
Visualization.visualize_spatiotemporal_granules(video[:5], "../results/spoon/test/spatiotemporal_granules_no_bbox_3.jpg", 3, False)
# Visualization.visualize_rgb_d_granules(video[:5], "../results/spoon/rgb_d_granules.jpg", 3)

def test_rule_base_and_graphs(video):
    threshold_clr = 50
    threshold_rgb = 15
    p = 3
    frames = video

    initial_frames = frames[:p]
    subsequent_frame = frames[p]

    diff_3D_matrix = compute_3D_difference_matrix(initial_frames, subsequent_frame)
    median_matrix = compute_median_matrix(diff_3D_matrix)
    med_threshold = 0.2 * np.max(median_matrix)
    med_threshold = 15

    # _gib bo ktotka z granules(g), initial_colors(i) i bounding_boxes(b)
    spatio_temporal_gib = form_spatiotemporal_granules(diff_3D_matrix, med_threshold)

    rgb_gib = form_rgb_d_granules(spatio_temporal_gib[0], spatio_temporal_gib[1], spatio_temporal_gib[2], threshold_rgb)

    # tylko do testow
    d_gib = create_granules_color(median_matrix, threshold_rgb)
    # d_gib = rgb_gib

    subsequent_spatio_colour_gib = create_granules_color(subsequent_frame, threshold_clr)

    rule_base, features = generate_rule_base(subsequent_spatio_colour_gib, spatio_temporal_gib, rgb_gib, d_gib)

    rule_base_scaled = (rule_base / 2 * 255).astype(np.uint8)
    cv2.imwrite("../results/spoon/rule_base_5.jpg", rule_base_scaled)

    flow_graph = generate_flow_graph(features)
    Visualization.draw_flow_graph(flow_graph)

    current_frame = frames[p + 1]
    current_spatio_colour_granules = create_granules_color(current_frame, threshold_clr)
    current_rule_base, current_features = generate_rule_base(current_spatio_colour_granules, spatio_temporal_gib,
                                                             rgb_gib, d_gib)
    coverage = compute_rule_base_coverage(flow_graph, current_features)  # jak cos sie wykrzaczy to tutaj
    print(coverage)


# cos sie liczy i jakis wynik wychodzi ale czy jest dobry to niewiadomo (trzeba sprawdzic jeszcze na poprawnych granulach glebokosciwych)
# test_rule_base_and_graphs(video)
# object_tracking(video, 50, 3)


