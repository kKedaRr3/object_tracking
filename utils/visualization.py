import numpy as np
from preprocessing import video_loader
from preprocessing import granulation
import cv2
from preprocessing.granulation import find_max_granule_index, form_rgb_d_granules
from preprocessing.video_preprocessing import compute_3D_difference_matrix, compute_median_matrix


class Visualization:

    @staticmethod
    def visualize_spatiotemporal_granules(frames, output_path, p, bbox=True, bbox_color=(255, 0, 0)):
        if len(frames) <= 3:
            raise Exception('Not enough frames to visualize')
        diff_frames = compute_3D_difference_matrix(frames[:p], frames[p])
        median_frame = compute_median_matrix(diff_frames)
        threshold = 0.3 * np.max(median_frame)

        granules, initial_colors, bounding_boxes = granulation.form_spatiotemporal_granules(diff_frames, threshold)

        if bbox:
            for granule_index, bbox in bounding_boxes.items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(frames[np.floor(p / 2)], (minX, minY), (maxX, maxY), bbox_color, 1)
            cv2.imwrite(output_path, frames[-1])
        else:
            result = np.zeros_like(frames[0])
            for y in range(frames[0].shape[0]):
                for x in range(frames[0].shape[1]):
                    if granules[y, x] is not None:
                        result[y, x] = initial_colors[granules[y, x]]
            cv2.imwrite(output_path, result)

    @staticmethod
    def visualize_spatio_color_granules(image, output_path, threshold=2, rgb=True, bbox=True, bbox_color=(0, 0, 0)):
        if type(image).__name__ == 'str':
            image = cv2.imread(image)

        if rgb:
            granules, initial_colors, bounding_boxes = granulation.create_granules_color(image, threshold)
        else:
            granules, initial_colors, bounding_boxes = granulation.create_granules_gray(image, threshold)

        if bbox:
            for granule_index, bbox in bounding_boxes.items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(image, (minX, minY), (maxX, maxY), bbox_color, 1)
            cv2.imwrite(output_path, image)

        else:
            result = np.zeros_like(image)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if granules[y, x] is not None:
                        result[y, x] = initial_colors[granules[y, x]]
            cv2.imwrite(output_path, result)

    @staticmethod
    def visualize_rgb_d_granules(frames, output_path, p, bbox=True, bbox_color=(255, 0, 0)):
        if len(frames) <= 3:
            raise Exception('Not enough frames to visualize')
        diff_frames = compute_3D_difference_matrix(frames[:p], frames[p])
        median_frame = compute_median_matrix(diff_frames)
        threshold = 0.3 * np.max(median_frame)

        granules, initial_colors, bounding_boxes = granulation.form_spatiotemporal_granules(diff_frames, threshold)
        rgb_granules, rgb_initial_colors, rgb_bounding_boxes = form_rgb_d_granules(granules, initial_colors, bounding_boxes, threshold)

        if bbox:
            for granule_index, bbox in rgb_bounding_boxes.items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(frames[0], (minX, minY), (maxX, maxY), bbox_color, 1)
            cv2.imwrite(output_path, frames[0])
        else:
            result = np.zeros_like(frames[0])
            for y in range(frames[0].shape[0]):
                for x in range(frames[0].shape[1]):
                    if granules[y, x] is not None:
                        result[y, x] = rgb_initial_colors[rgb_granules[y, x]]
            cv2.imwrite(output_path, result)