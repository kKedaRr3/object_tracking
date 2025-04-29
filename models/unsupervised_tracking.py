import cv2
import numpy as np

from models.flow_graph import generate_flow_graph, compute_rule_base_coverage, get_features_to_update
from postprocessing.frame_postprocessor import draw_tracked_object_bbox, create_video_from_frames
from preprocessing.granulation import form_spatiotemporal_granules, form_rgb_d_granules, create_granules_color
from preprocessing.video_preprocessing import compute_3D_difference_matrix, compute_median_matrix
from models.rulebase import generate_rule_base, segment_foreground
from utils.visualization import Visualization


def object_tracking(frames: np.array, depth_frames: np.array, output_path: str, threshold: int = 2, p: int = 3):
    """
    Główny algorytm śledzenia obiektów w klatkach wideo.
    """

    rb_threshold = threshold * 4

    initial_frames = frames[:p]
    subsequent_frame = frames[p]
    processed_frames = []

    diff_3D_matrix = compute_3D_difference_matrix(initial_frames, subsequent_frame)
    # median_matrix = compute_median_matrix(diff_3D_matrix)
    # med_threshold = 0.3 * np.max(median_matrix)


    spatio_temporal_gib = form_spatiotemporal_granules(diff_3D_matrix, rb_threshold)

    rgb_gib = form_rgb_d_granules(spatio_temporal_gib[0], spatio_temporal_gib[1], spatio_temporal_gib[2], rb_threshold)

    initial_depth_frames, subsequent_depth_frame = depth_frames[:p], depth_frames[p]
    depth_diff_3D_matrix = compute_3D_difference_matrix(initial_depth_frames, subsequent_depth_frame)
    d_sp_t_gib = form_spatiotemporal_granules(depth_diff_3D_matrix, rb_threshold)
    d_gib = form_rgb_d_granules(d_sp_t_gib[0], d_sp_t_gib[1], d_sp_t_gib[2], rb_threshold)

    subsequent_spatio_colour_gib = create_granules_color(subsequent_frame, threshold)

    rule_base, features = generate_rule_base(subsequent_spatio_colour_gib, spatio_temporal_gib, rgb_gib, d_gib)

    flow_graph = generate_flow_graph(features)

    foreground = segment_foreground(rule_base)

    object_bbox = track_object(foreground)

    processed_frames.append(draw_tracked_object_bbox(object_bbox, subsequent_frame))

    for frame_index in range(p + 1, len(frames)):
        print(f"\nframe {frame_index}/{len(frames)}")
        current_frame = frames[frame_index]

        current_spatio_colour_gib = create_granules_color(current_frame, threshold)

        rule_base, features = generate_rule_base(current_spatio_colour_gib, spatio_temporal_gib, rgb_gib, d_gib)

        '''for tests'''
        rule_base_scaled = (rule_base / 2 * 255).astype(np.uint8)
        cv2.imwrite(f"../results/man/rule_base/test/frame_{frame_index}.jpg", rule_base_scaled)
        '''for tests'''

        coverage, test_flow_graph = compute_rule_base_coverage(flow_graph, features)
        print("coverage: ", coverage)

        # 0.0002 Troche za male i za kazdym razem sie aktualizuje
        if coverage > 0.0005:
            features_to_update = get_features_to_update(flow_graph, test_flow_graph, 0.4)
            print("\n\nupdate required")
            print(features_to_update)
            prev_frames = frames[frame_index - p: frame_index]
            if "sp_t" in features_to_update or "rgb" in features_to_update:
                diff_3D_matrix = compute_3D_difference_matrix(prev_frames, current_frame)
                spatio_temporal_gib = form_spatiotemporal_granules(diff_3D_matrix, rb_threshold)
                if "rgb" in features_to_update:
                    rgb_gib = form_rgb_d_granules(spatio_temporal_gib[0], spatio_temporal_gib[1],
                                                  spatio_temporal_gib[2], rb_threshold)

            if "d" in features_to_update:
                prev_depth_frames = depth_frames[frame_index - p: frame_index]
                current_depth_frame = depth_frames[frame_index]
                depth_diff_3D_matrix = compute_3D_difference_matrix(prev_depth_frames, current_depth_frame)
                d_sp_t_bib = form_spatiotemporal_granules(depth_diff_3D_matrix, rb_threshold)
                d_gib = form_rgb_d_granules(d_sp_t_bib[0], d_sp_t_bib[1], d_sp_t_bib[2], rb_threshold)

            rule_base, features = generate_rule_base(current_spatio_colour_gib, spatio_temporal_gib, rgb_gib,
                                                     d_gib)
            flow_graph = generate_flow_graph(features)

        # Visualization.draw_flow_graph(flow_graph)

        foreground = segment_foreground(rule_base)

        object_bbox = track_object(foreground)

        print("\n\n", object_bbox, "\n\n")

        processed_frames.append(draw_tracked_object_bbox(object_bbox, current_frame))

    create_video_from_frames(processed_frames, output_path)


def track_object(foreground):
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    minX = minY = float('inf')
    maxX = maxY = -float('inf')

    for point in largest_contour:
        x, y = point[0]

        minX = min(minX, x)
        minY = min(minY, y)
        maxX = max(maxX, x)
        maxY = max(maxY, y)

    return minY, minX, maxY, maxX
