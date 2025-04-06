import cv2

def convert_to_grayscale(frame):
    """
    Zwraca kopię frame w skali szarości (1 kanał).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def batch_grayscale(frames):
    """
    Dla listy klatek (frames) konwertuje każdą na skalę szarości.
    Zwraca listę nowych klatek.
    """
    gray_frames = []
    for f in frames:
        gray_frames.append(convert_to_grayscale(f))
    return gray_frames
