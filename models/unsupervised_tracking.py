import numpy as np

from preprocessing import video_preprocessing
from preprocessing.granulation import form_spatiotemporal_granules, create_granules, form_rgb_d_granules
from preprocessing.temporal_segmentation import three_point_approximation
from preprocessing.video_preprocessing import compute_3D_difference_matrix, compute_median_matrix
from models.rulebase import RoughRuleBase, initialize_rule_base

'''
Tak ma wygaldac koncowy algorytm unsupervised object tracking
'''
def object_tracking(frames, threshold=2, p=3):
    """
    Główny algorytm śledzenia obiektów w klatkach wideo.

    zrobione przetestowane i jakos wygladajace:

    """

    # 1. Inicjalizacja
    initial_frames = frames[:p]  # Pierwsze p klatek
    subsequent_frame = frames[p]
    rule_base = None  # Początkowa rule-base

    diff_3D_matrix = compute_3D_difference_matrix(initial_frames, subsequent_frame)
    median_matrix = compute_median_matrix(diff_3D_matrix)
    med_threshold = 0.3 * np.max(median_matrix)

    spatio_temporal_granules, sp_t_initial_colors, sp_t_bounding_boxes = form_spatiotemporal_granules(diff_3D_matrix, med_threshold)

    rgb_granules = form_rgb_d_granules(spatio_temporal_granules, sp_t_initial_colors, sp_t_bounding_boxes, threshold)

    depth_median_matrix = None # TODO Pobrac jakies wideo ktore ma 4 kanaly RGBD
    d_granules = create_granules(depth_median_matrix, threshold)

    #do porownywania z sp_t rgb i d granules w celu utworzenia bazy regul
    subsequent_spatio_colour_granules = create_granules(subsequent_frame, threshold)

    # 3. Obliczanie Rule-base
    rule_base = initialize_rule_base(subsequent_spatio_colour_granules, spatio_temporal_granules, rgb_granules, d_granules) # TODO

    # 4. Generowanie flow-grapha dla kolejnej klatki
    flow_graph = generate_initial_flow_graph(subsequent_spatio_colour_granules, rule_base) # TODO

    # 5. Segmentacja tła i obiektu
    foreground = segment_foreground(flow_graph, rule_base) # TODO

    # 6. Śledzenie obiektu
    track_object(foreground) # TODO

    # 7. Kolejne iteracje (Algorytm 13)
    for frame_index in range(p + 1, len(frames)):
        prev_frames = frames[frame_index - p: frame_index]
        current_frame = frames[frame_index]

        # 7.1. Generowanie nowych granulek spatio-color
        current_spatio_colour_granules = create_granules(current_frame, threshold)

        # 7.2. Ocena pokrycia rule-base za pomocą flow-grapha
        coverage = compute_rule_base_coverage(flow_graph, rule_base)

        # 7.3. Sprawdzanie wariancji i ewentualna aktualizacja rule-base
        if coverage_variance_is_significant(coverage):  # Zakładając, że masz funkcję do tego
            rule_base = update_rule_base(prev_frames, rule_base)



        # 7.4. Segmentacja tła i obiektu w nowej klatce
        foreground = segment_foreground(flow_graph, rule_base)

        # 7.5. Śledzenie obiektu
        track_object(foreground)