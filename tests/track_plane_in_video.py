import cv2
import numpy as np
import os


#############################
# KROK 1: Funkcje pomocnicze
#############################

def three_point_approximation(prev_frames_gray, current_gray, e=3.0):
    """
    prev_frames_gray: lista 3 poprzednich klatek w skali szarości.
    current_gray: bieżąca klatka w skali szarości.
    Zwraca maskę binarną (0/1) obszaru ruchu.
    """
    f1, f2, f3 = prev_frames_gray
    optimistic = np.max(np.stack([f1, f2, f3]), axis=0)
    pessimistic = np.min(np.stack([f1, f2, f3]), axis=0)
    median_val = np.median(np.stack([f1, f2, f3]), axis=0)

    mean_est = (optimistic + 4 * median_val + pessimistic) / 6.0
    std_est = (optimistic - pessimistic) / 6.0

    diff = np.abs(current_gray - mean_est)
    mask = (diff > (e * std_est)).astype(np.uint8)
    return mask


def clean_noise(mask, kernel_size=5):
    """
    Usuwa drobny szum metodą morfologiczną (open).
    Zwraca oczyszczoną maskę 0/1.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # convert to 0/1
    opened_bin = (opened > 0).astype(np.uint8)
    return opened_bin


def find_largest_contour_bbox(mask):
    """
    Znajduje największy kontur w masce (0/1 lub 0/255).
    Zwraca (x, y, w, h) lub None, jeśli brak konturów.
    """
    # Upewnij się, że maska jest w formacie 0/255:
    mask_255 = mask * 255 if mask.max() == 1 else mask.copy()

    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)


#############################
# KROK 2: Główny pipeline
#############################

def track_plane_in_video(input_video, output_video, e=3.0):
    """
    Odczytuje video z input_video, dla każdej klatki wykrywa ruch metodą
    three-point approximation, rysuje bounding box na największym obszarze ruchu
    i zapisuje do output_video.
    """

    # Otwarcie wideo źródłowego
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Nie udało się otworzyć pliku: {input_video}")
        return

    # Pobranie parametrów oryginalnego filmu
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Kodowanie wideo (MP4 - 'mp4v', albo 'XVID', 'MJPG' itd.)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Utworzenie obiektu VideoWriter do zapisu
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Przechowywanie 3 poprzednich klatek w skali szarości
    prev_frames_gray = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # koniec wideo

        # Konwersja do szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dodaj bieżącą klatkę do bufora
        if len(prev_frames_gray) < 3:
            prev_frames_gray.append(gray)
            # Jeszcze nie mamy 3 klatek wstecz, więc nie robimy detekcji
            out.write(frame)
        else:
            # Mamy już 3 poprzednie klatki
            mask = three_point_approximation(prev_frames_gray, gray, e=e)
            mask_clean = clean_noise(mask, kernel_size=5)

            bbox = find_largest_contour_bbox(mask_clean)
            # Rysuj bounding box, jeśli coś wykryto
            if bbox is not None:
                (x, y, w, h) = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Zapis do pliku wideo
            out.write(frame)

            # Aktualizujemy bufor klatek
            prev_frames_gray.pop(0)  # usuń najstarszą
            prev_frames_gray.append(gray)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Zapisano wynik do pliku: {output_video}")


#############################
# KROK 3: Uruchom test
#############################

if __name__ == "__main__":
    input_vid = "../data/F1.mp4"  # Ścieżka do Twojego pliku wejściowego
    output_vid = "samolot_wynik4.mp4"  # Gdzie zapisać efekt

    track_plane_in_video(input_vid, output_vid, e=3.0)
