import cv2
import numpy as np

def normalize_frame_0_1(frame):
    """
    Normalizuje wartości pikseli tak, by były w zakresie [0, 1].
    frame może być w BGR lub w skali szarości.
    Zwraca numpy array (float32).
    """
    normalized = frame.astype(np.float32) / 255.0
    return normalized

def normalize_frame_0_255(frame):
    """
    Normalizuje wartości pikseli do zakresu [0,255].
    Zakładamy, że frame jest float64 lub float32.
    Zwraca numpy array (uint8).
    """
    if frame.dtype not in [np.float32, np.float64]:
        frame = frame.astype(np.float32)  # w razie potrzeby
    norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)
