import cv2
import os
import numpy as np

from preprocessing.frame_loader import load_frames_from_folder
from preprocessing.background_subtraction import three_point_approximation
from preprocessing.mask_utils import clean_noise


def find_largest_contour_bbox(binary_mask):
    """
    Znajduje największy kontur w masce, zwraca (x, y, w, h).
    Jeśli brak konturów - zwraca None.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)


def run_automatic_plane_detection(frames):
    """
    Automatyczne wykrywanie samolotu w każdej klatce przy użyciu
    three_point_approximation + morfologii.
    Zwraca listę bounding boxów (x, y, w, h) dla kolejnych klatek.
    Jeśli w danej klatce nie ma konturu, dajemy None.
    """
    # Ilość klatek = len(frames)
    # Wymagamy co najmniej 3 klatek wstecz, więc zaczynamy od klatki nr 3
    bboxes = []
    for i in range(len(frames)):
        if i < 3:
            # Za mało klatek do three_point_approximation
            bboxes.append(None)
            continue

        # Pobieramy 3 poprzednie klatki - w skali szarości (załóżmy, że frames[i] jest grayscale)
        # Jeśli frames to obrazy BGR, trzeba przekonwertować do szarości
        f1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frames[i - 2], cv2.COLOR_BGR2GRAY)
        f3 = cv2.cvtColor(frames[i - 3], cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # 1) Generuj maskę
        prevs = [f1, f2, f3]
        mask = three_point_approximation(prevs, current_gray, e=3.0)

        # 2) Oczyszczanie maski (np. open)
        mask_clean = clean_noise(mask, kernel_size=5)  # mask_utils.py

        # 3) Znajdź bounding box największego obszaru
        bbox = find_largest_contour_bbox(mask_clean)
        bboxes.append(bbox)

    return bboxes


def main():
    # 1) Wczytanie klatek z folderu "data/"
    folder = "../data/"
    frames = load_frames_from_folder(folder, max_frames=30)  # np. 30 klatek

    if len(frames) < 4:
        print("Za mało klatek, minimum 4!")
        return

    # 2) Automatyczna detekcja samolotu
    bboxes = run_automatic_plane_detection(frames)

    # 3) Rysowanie i zapisanie wyników
    os.makedirs("auto_plane_results", exist_ok=True)
    for i, bbox in enumerate(bboxes):
        frame_vis = frames[i].copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        out_path = os.path.join("auto_plane_results", f"plane_result_{i}.jpg")
        cv2.imwrite(out_path, frame_vis)

    print("Gotowe! Sprawdź folder 'auto_plane_results'.")


if __name__ == "__main__":
    main()
