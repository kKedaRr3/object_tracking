import numpy as np

from preprocessing import video_loader
from preprocessing import granulation
import cv2

from preprocessing.granulation import find_max_granule_index


class Visualization:

    @staticmethod
    def visualize_video_granulation(frames, output_path, threshold=2, color=(0, 0, 0)):
        if type(frames).__name__ == 'str':
            frames = video_loader.load_frames_from_mp4(frames)
        processed_frames = granulation.form_spatiotemporal_granules(frames, threshold)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                              (frames[0].shape[1], frames[0].shape[0]))


        for frame_index in range(len(frames)):
            frame = frames[frame_index]
            bounding_boxes = processed_frames[frame_index][2]
            for granule_index, bbox in bounding_boxes.items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), color, 1)

            out.write(frame)
        out.release()


    @staticmethod
    def visualize_image_granulation(image, output_path, threshold=2, rgb=True, bbox=True, bbox_color=(0, 0, 0)):
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



