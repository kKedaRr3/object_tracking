import numpy as np
import cv2


def compute_3D_difference_matrix(frames, current_frame):
    difference = []
    for frame in frames:
        diff_image = cv2.absdiff(frame, current_frame)
        diff_image = np.clip(diff_image, 0, 255).astype(np.uint8)
        difference.append(diff_image)

    return difference

def compute_median_matrix(difference_3D_matrix):
    height, width, _ = difference_3D_matrix[0].shape

    median_matrix = np.zeros((height, width, 3))  # 3 kanały RGB

    for y in range(height):
        for x in range(width):
            pixel_values_r = [difference_3D_matrix[i][y, x, 0] for i in range(len(difference_3D_matrix))]
            pixel_values_g = [difference_3D_matrix[i][y, x, 1] for i in range(len(difference_3D_matrix))]
            pixel_values_b = [difference_3D_matrix[i][y, x, 2] for i in range(len(difference_3D_matrix))]

            median_matrix[y, x, 0] = np.median(pixel_values_r)  # Red channel
            median_matrix[y, x, 1] = np.median(pixel_values_g)  # Green channel
            median_matrix[y, x, 2] = np.median(pixel_values_b)  # Blue channel

    return median_matrix
