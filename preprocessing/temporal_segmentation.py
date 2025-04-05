import cv2
import numpy as np

def three_point_approximation(frames_3, current_frame, e=3.0):
    """
    frames_3: lista 3 poprzednich klatek w skali szarości
    current_frame: obecna klatka (także w szarości)
    e=3.0 – współczynnik do wyznaczania progu
    Zwraca maskę binarną (1 = obiekt, 0 = tło).
    """
    for i in range(len(frames_3)):
        frames_3[i] = cv2.cvtColor(frames_3[i], cv2.COLOR_RGB2GRAY)
        frames_3[i] = cv2.medianBlur(frames_3[i], 5)
        cv2.imwrite(f"../results/a_test_{i}.jpg", frames_3[i])
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    f1, f2, f3 = frames_3  # 3 poprzednie klatki
    optimistic = np.max(np.stack([f1, f2, f3]), axis=0)
    median_val = np.median(np.stack([f1, f2, f3]), axis=0)
    pessimistic = np.min(np.stack([f1, f2, f3]), axis=0)

    # approx mean i approx std
    mean_est = (optimistic + 4 * median_val + pessimistic) / 6.0
    std_est = (optimistic - pessimistic) / 6.0

    diff = np.abs(current_frame - mean_est)

    mask = (diff > e * std_est).astype(np.uint8) * 255

    return mask