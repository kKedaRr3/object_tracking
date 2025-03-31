import cv2
import os


def video_to_frames(video_path, output_folder):
    """
    Wczytuje wideo z pliku video_path i zapisuje je klatka po klatce
    w formacie JPG do folderu output_folder.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Nie można otworzyć pliku wideo: {video_path}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # koniec wideo
        out_name = os.path.join(output_folder, f"frame{frame_count}.jpg")
        cv2.imwrite(out_name, frame)
        frame_count += 1

    cap.release()
    print(f"Zapisano {frame_count} klatek do folderu: {output_folder}")
