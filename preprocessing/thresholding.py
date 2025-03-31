import cv2
import numpy as np

def simple_threshold(frame_gray, thresh_val=127):
    """
    Nakłada próg thresh_val na obraz w skali szarości.
    Zwraca maskę binarną (0/255).
    """
    _, bin_mask = cv2.threshold(frame_gray, thresh_val, 255, cv2.THRESH_BINARY)
    return bin_mask

def otsu_threshold(frame_gray):
    """
    Binaryzacja metodą Otsu w OpenCV.
    Zwraca maskę binarną (0/255).
    """
    _, bin_mask = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_mask

def adaptive_threshold(frame_gray, block_size=11, c=2):
    """
    Binaryzacja adaptacyjna - sprawdza okolice piksela.
    block_size musi być nieparzyste.
    c - stała odejmowana od średniej
    """
    bin_mask = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block_size, c)
    return bin_mask
