from preprocessing import video_processing
from preprocessing import granulation
import cv2


class Visualization:

    @staticmethod
    def visualize_video_granulation(input_path, output_path):
        frames = video_processing.load_frames_from_mp4(input_path)
        granules, initial_colors, bounding_boxes = granulation.form_spatiotemporal_granules(frames, 2, 3)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                              (frames[0].shape[1], frames[0].shape[0]))

        for frame_index in range(len(frames) - 1):
            frame = frames[frame_index]
            for granule_index, bbox in bounding_boxes[frame_index].items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 0), 1)

            out.write(frame)

        out.release()

    @staticmethod
    def visualize_image_granulation(input_path, output_path):
        image = cv2.imread(input_path)
        granules, initial_colors, bounding_boxes = granulation.create_granules(image, 2)

        for granule_index, bbox in bounding_boxes.items():
            minY, minX, maxY, maxX = bbox
            cv2.rectangle(image, (minX, minY), (maxX, maxY), (0, 0, 0), 1)

        cv2.imwrite(output_path, image)
