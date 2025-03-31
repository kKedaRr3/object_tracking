import numpy as np
from models.segmentation import compute_spatial_segmentation, three_point_approximation


def run_partial_supervised_tracking(frames, initial_bbox, base=10, e=3.0):
    """
    frames        - lista klatek (np. numpy array)
    initial_bbox  - (x, y, w, h) początkowa pozycja obiektu
    Zwraca np. listę bbox na każdą klatkę.
    """
    # Załóżmy, że mamy co najmniej 4 klatki
    track_bboxes = []
    track_bboxes.append(initial_bbox)

    # Inicjalizacja
    # ...
    for i in range(3, len(frames)):
        prev_frames = [frames[i-1], frames[i-2], frames[i-3]]
        current_frame = frames[i]

        # 1) Spatial segmentation
        # Tu ewentualnie quadtree granules
        # ...
        granules = None  # generuj lub wczytaj
        spatial_mask, best_thresh = compute_spatial_segmentation(current_frame, granules, base=base)

        # 2) Temporal segmentation
        temporal_mask = three_point_approximation(prev_frames, current_frame, e=e)

        # 3) Połączenie wyników -> maska finalna
        combined_mask = np.logical_or(spatial_mask, temporal_mask).astype(np.uint8)

        # 4) Aktualizacja trackera (np. bounding box) na podstawie maski wewnątrz poprzedniego bbox
        #    W praktyce: wycinamy obszar z combined_mask, znajdujemy największe skupisko pikseli...
        x, y, w, h = track_bboxes[-1]
        roi = combined_mask[y:y+h, x:x+w]
        # Prosty przykład: jeśli pikseli "1" jest mało, to tracker się przesuwa o 1 w bok...
        # W praktyce: centroid, boundingRect, itp.
        new_x = x
        new_y = y
        # ...
        new_bbox = (new_x, new_y, w, h)
        track_bboxes.append(new_bbox)

    return track_bboxes
