import os
import cv2
import numpy as np

# Importy z folderu preprocessing:
from preprocessing.frame_extractor import video_to_frames
from preprocessing.frame_loader import load_frames_from_folder
from preprocessing.grayscale import convert_to_grayscale
from preprocessing.temporal_segmentation import three_point_approximation
from preprocessing.thresholding import simple_threshold, otsu_threshold, adaptive_threshold

def test_preprocessing_pipeline():
    """
    Przykładowy test łączący kilka funkcji z folderu `preprocessing/`.
    """

    # 1. Rozdzielenie wideo na klatki
    video_path = "../data/test.mp4"   # zakładamy, że uruchamiasz z folderu `tests/`
    output_folder = "../data/frames_test/"    # folder na klatki
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("==> Ekstrakcja klatek z wideo:")
    video_to_frames(video_path, output_folder)

    # 2. Wczytanie kilku klatek (np. do 5)
    print("==> Wczytywanie klatek:")
    frames = load_frames_from_folder(output_folder, max_frames=5)
    print(f"Wczytano {len(frames)} klatek.")

    if len(frames) == 0:
        print("Brak klatek do przetworzenia – sprawdź ścieżki!")
        return

    # 3. Konwersja do skali szarości (jako przykład)
    print("==> Konwersja do skali szarości:")
    gray_frames = [convert_to_grayscale(f) for f in frames]

    # 4. Odejomowanie tła metodą Three-point Approx (potrzebujemy 3 poprzednich klatek)
    #    Na potrzeby testu załóżmy, że mamy co najmniej 4 klatki:
    if len(gray_frames) >= 4:
        # weźmy 3 poprzednie + obecną
        f1, f2, f3 = gray_frames[:3]
        current = gray_frames[3]
        mask_moving = three_point_approximation([f1, f2, f3], current, e=3.0)

        # Zapisz wynik do pliku, żeby zobaczyć
        cv2.imwrite("moving_mask.jpg", mask_moving * 255)  # maska 0/1 → 0/255
        print("Zapisano moving_mask.jpg")

    # 5. Thresholding (np. Otsu) na pierwszej klatce w skali szarości
    if len(gray_frames) > 0:
        otsu_mask = otsu_threshold(gray_frames[0])
        cv2.imwrite("otsu_result.jpg", otsu_mask)
        print("Zapisano otsu_result.jpg (binaryzacja metodą Otsu)")

    print("==> Zakończono test przetwarzania wstępnego.")

if __name__ == "__main__":
    test_preprocessing_pipeline()
