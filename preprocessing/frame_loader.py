import cv2
import os

def load_frames_from_folder(folder_path, max_frames=None):
    """
    Wczytuje wszystkie (lub max_frames) pliki .jpg z folderu folder_path,
    zwraca listę klatek (numpy array).
    """
    frames = []
    files = sorted(os.listdir(folder_path))
    count = 0
    for fname in files:
        if max_frames is not None and count >= max_frames:
            break
        if fname.lower().endswith((".jpg", ".png")):
            fpath = os.path.join(folder_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)  # wczytaj w kolorze
            if img is not None:
                frames.append(img)
                count += 1
    return frames
