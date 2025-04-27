import numpy as np
import cv2


def compute_3D_difference_matrix(frames, current_frame):
    difference = []
    for frame in frames:
        diff_image = cv2.absdiff(frame, current_frame)
        diff_image = np.clip(diff_image, 0, 255).astype(np.uint8)
        difference.append(diff_image)

    return erode_diff_frames(difference)


def compute_median_matrix(difference_3D_matrix):
    height, width, _ = difference_3D_matrix[0].shape

    median_matrix = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            pixel_values_r = [difference_3D_matrix[i][y, x, 0] for i in range(len(difference_3D_matrix))]
            pixel_values_g = [difference_3D_matrix[i][y, x, 1] for i in range(len(difference_3D_matrix))]
            pixel_values_b = [difference_3D_matrix[i][y, x, 2] for i in range(len(difference_3D_matrix))]

            median_matrix[y, x, 0] = np.median(pixel_values_r)
            median_matrix[y, x, 1] = np.median(pixel_values_g)
            median_matrix[y, x, 2] = np.median(pixel_values_b)

    return median_matrix


def median_filtration(diff_frames):
    for index, frame in enumerate(diff_frames):
        diff_frames[index] = cv2.medianBlur(frame, 11)
    return diff_frames


def erode_diff_frames(diff_frames):
    kernel = np.ones((5, 5), np.uint8)
    for index, frame in enumerate(diff_frames):
        diff_frames[index] = cv2.erode(frame, kernel)
    return diff_frames


def histogram_equalization(diff_frames):
    for index, frame in enumerate(diff_frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])
        diff_frames[index] = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    return diff_frames
