import cv2
import numpy as np

def morphological_close(mask, kernel_size=5):
    """
    Morfologiczne 'close' w celu zasklepienia dziur w masce.
    Zwraca maskę 0/1.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (closed > 0).astype(np.uint8) * 255

def morphological_open(mask, kernel_size=5):
    """
    Morfologiczne 'open' w celu usunięcia drobnego szumu.
    Zwraca maskę 0/1.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (opened > 0).astype(np.uint8) * 255

def combine_morph(mask, open_size=3, close_size=5):
    """
    Najpierw 'open' a potem 'close', często używane w analizie obiektów.
    """
    temp = morphological_open(mask, kernel_size=open_size)
    result = morphological_close(temp, kernel_size=close_size)
    return result * 255

def find_largest_contour_bbox(mask):
    """
    Znajduje największy kontur w masce (0/1 lub 0/255).
    Zwraca (x, y, w, h) lub None, jeśli brak konturów.
    """
    if mask.max() == 1:
        mask_255 = (mask * 255).astype(np.uint8)
    else:
        mask_255 = mask.copy()

    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)
