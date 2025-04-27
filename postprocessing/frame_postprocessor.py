import cv2


def draw_tracked_object_bbox(object_bbox, frame):
    minY, minX, maxY, maxX = object_bbox
    cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 1)

    return frame

def create_video_from_frames(frames, output_path, fps=30):
    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
