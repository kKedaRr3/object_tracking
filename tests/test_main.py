import os
import cv2
import numpy as np

# Importy z Twoich modułów:
from models.partial_supervised_tracking import run_partial_supervised_tracking
from models.multi_object_tracking import run_multiobject_tracking
# ewentualnie inne moduły:
# from models.segmentation import compute_spatial_segmentation
# from models.quadtree import quadtree_decomposition
# etc.

def load_frames_from_folder(folder_path, max_frames=10):
    """
    Wczytuje do max_frames obrazów z folderu i zwraca jako listę numpy array.
    """
    frames = []
    files = sorted(os.listdir(folder_path))
    for fname in files[:max_frames]:
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
    return frames

def test_partial_supervised_tracking():
    """
    Test działania algorytmu 'Partial Supervised Tracking'.
    Zakładamy, że mamy kilka klatek (np. 5-10),
    a w pierwszej klatce zdefiniujemy bounding box "ręcznie".
    """
    folder_path = "../data"  # dostosuj ścieżkę do obrazów
    frames = load_frames_from_folder(folder_path, max_frames=10)

    if len(frames) < 4:
        print("Za mało klatek do testu!")
        return

    # Zakładamy, że obiekt to np. fragment w lewym-górnym rogu
    # w 1. klatce:
    initial_bbox = (50, 50, 100, 100)  # (x, y, w, h) – hipotetyczna pozycja obiektu

    bboxes = run_partial_supervised_tracking(frames, initial_bbox)
    # Zwracamy listę bboxes – bounding boxów w kolejnych klatkach

    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        # Zwizualizujmy na obrazie
        frame_vis = frames[i].copy()
        cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
        out_name = f"partial_track_result_{i}.jpg"
        cv2.imwrite(out_name, frame_vis)
    print("Zapisano wyniki do plików partial_track_result_*.jpg")

def test_multiobject_tracking():
    """
    Test działania algorytmu 'Unsupervised Multiple Object Tracking'.
    """
    folder_path = "../data"
    frames = load_frames_from_folder(folder_path, max_frames=10)

    if len(frames) < 5:
        print("Za mało klatek do testu!")
        return

    # Wywołaj funkcję z multi_object_tracking.py
    objects_positions = run_multiobject_tracking(frames, P=3)

    # Tutaj 'objects_positions' to przykładowa lista z koordynatami
    # lub inną formą opisującą obiekty w każdej klatce.
    # W prostym wariancie mogłoby to być:
    #   [ [ (x1,y1,w1,h1), (x2,y2,w2,h2) ],  # obiekty w klatce 0
    #     [ (x1,y1,w1,h1), (x2,y2,w2,h2) ],  # obiekty w klatce 1
    #     ...
    #   ]
    # Musisz to dostosować do faktycznego zwracanego formatu.

    for i, frame_objs in enumerate(objects_positions):
        frame_vis = frames[i].copy()
        for obj_bbox in frame_objs:
            x, y, w, h = obj_bbox
            cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
        out_name = f"multi_track_result_{i}.jpg"
        cv2.imwrite(out_name, frame_vis)

    print("Zapisano wyniki do plików multi_track_result_*.jpg")

if __name__ == "__main__":
    print("== Test Partial Supervised Tracking ==")
    test_partial_supervised_tracking()
    print("== Test Multi-Object Tracking ==")
    test_multiobject_tracking()
