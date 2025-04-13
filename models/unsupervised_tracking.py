import numpy as np
from preprocessing.granulation import form_spatiotemporal_granules, form_rgb_d_granules, \
    create_granules_color
from preprocessing.video_preprocessing import compute_3D_difference_matrix, compute_median_matrix
from models.rulebase import generate_rule_base


def object_tracking(frames, threshold=2, p=3):
    """
    Główny algorytm śledzenia obiektów w klatkach wideo.
    """

    initial_frames = frames[:p]
    subsequent_frame = frames[p]

    diff_3D_matrix = compute_3D_difference_matrix(initial_frames, subsequent_frame)
    median_matrix = compute_median_matrix(diff_3D_matrix)
    med_threshold = 0.3 * np.max(median_matrix)

    # _gib bo ktotka z granules(g), initial_colors(i) i bounding_boxes(b)
    spatio_temporal_gib = form_spatiotemporal_granules(diff_3D_matrix, med_threshold)

    rgb_gib = form_rgb_d_granules(spatio_temporal_gib[0], spatio_temporal_gib[1], spatio_temporal_gib[2], threshold)

    initial_depth_frames, subsequent_depth_frame = None, None  # TODO Pobrac jakies wideo ktore ma 4 kanaly RGBD
    depth_diff_3D_matrix = compute_3D_difference_matrix(initial_depth_frames, subsequent_depth_frame)
    depth_median_matrix = compute_median_matrix(depth_diff_3D_matrix)
    d_gib = create_granules_color(depth_median_matrix, threshold)

    # do porownywania z sp_t rgb i d granules w celu utworzenia bazy regul
    subsequent_spatio_colour_gib = create_granules_color(subsequent_frame, threshold)

    rule_base = generate_rule_base(subsequent_spatio_colour_gib, spatio_temporal_gib, rgb_gib, d_gib)

    flow_graph = generate_flow_graph(rule_base)  # TODO

    foreground = segment_foreground(rule_base)  # TODO

    track_object(foreground)  # TODO

    for frame_index in range(p + 1, len(frames)):
        current_frame = frames[frame_index]

        current_spatio_colour_granules = create_granules(current_frame, threshold)

        rule_base = generate_rule_base(current_spatio_colour_granules, spatio_temporal_gib, rgb_gib, d_gib)

        coverage = compute_rule_base_coverage(flow_graph, rule_base)
        #TODO
        # tutaj flow_graph to bedzie ten treningowy a z rule_base trzbea
        # bedzie zrobic testowy czyli ten wynikajacy ze stanu faktycznego
        # flow_graph jest to jen z wczesniejszej klatki

        if not coverage_variance_is_significant(coverage):
            features_to_update = get_features_to_update()  # TODO
            prev_frames = frames[frame_index - p: frame_index]
            if "sp_t" or "rgb" in features_to_update:
                diff_3D_matrix = compute_3D_difference_matrix(prev_frames, subsequent_frame)
                median_matrix = compute_median_matrix(diff_3D_matrix)
                med_threshold = 0.3 * np.max(median_matrix)
                spatio_temporal_gib = form_spatiotemporal_granules(diff_3D_matrix, med_threshold)
                if "rgb" in features_to_update:
                    rgb_gib = form_rgb_d_granules(spatio_temporal_gib[0], spatio_temporal_gib[1],
                                                  spatio_temporal_gib[2], threshold)

            if "d" in features_to_update:
                prev_depth_frames = None  # TODO
                depth_diff_3D_matrix = compute_3D_difference_matrix(prev_depth_frames[:-1], prev_depth_frames[-1])
                depth_median_matrix = compute_median_matrix(depth_diff_3D_matrix)
                d_gib = create_granules_color(depth_median_matrix, threshold)

            rule_base = generate_rule_base(current_spatio_colour_granules, spatio_temporal_gib, rgb_gib, d_gib)
            flow_graph = generate_flow_graph(rule_base)  # TODO

        foreground = segment_foreground(rule_base) # TODO

        track_object(foreground) # TODO
