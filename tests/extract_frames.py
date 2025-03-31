import cv2
import os


def video_to_frames(video_path, output_folder):
    # Otwórz wideo
    cap = cv2.VideoCapture(video_path)

    # Sprawdź, czy wideo jest otwarte
    if not cap.isOpened():
        print("Nie udało się otworzyć wideo")
        return

    # Utwórz folder, jeśli nie istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Zapisz klatkę jako obraz
        frame_filename = os.path.join(output_folder, f"frame{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Zapisano {frame_count} klatek wideo do folderu {output_folder}")


# Użycie funkcji
video_path = "../data/test.mp4"
output_folder = "../data"
video_to_frames(video_path, output_folder)
