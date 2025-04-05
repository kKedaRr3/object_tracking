from preprocessing import video_loader
from preprocessing import granulation
import cv2


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
    def visualize_image_granulation(image, output_path, threshold=2, color=(0, 0, 0)):
        if type(image).__name__ == 'str':
            image = cv2.imread(image)
        granules, initial_colors, bounding_boxes = granulation.create_granules(image, threshold)

        for granule_index, bbox in bounding_boxes.items():
            minY, minX, maxY, maxX = bbox
            cv2.rectangle(image, (minX, minY), (maxX, maxY), color, 1)

        cv2.imwrite(output_path, image)
