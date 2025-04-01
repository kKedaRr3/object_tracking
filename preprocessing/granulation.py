import numpy as np

import video_processing
import cv2

'''Funkcja do sprawdzania podobiensta kolorow'''
def colour_nearness(color1, color2, threshold):
    distance = np.linalg.norm(np.array(color1) - np.array(color2))
    return distance < threshold

'''Funkcja tworzaca granule'''
def create_granules(image, threshold):
    height, width = image.shape[:2]
    granules = [[[] for _ in range(width)] for _ in range(height)]
    initial_colors = dict()
    bounding_boxes = dict()
    granule_index = 0

    for y in range(height):
        for x in range(width):
            if np.all(image[y][x] == 0) or granules[y][x] != []:  # Pomijamy czarne piksele i już przypisane
                continue
            initial_colors[granule_index] = image[y][x]
            bounding_boxes[granule_index] = [y, x, y, x]  # [minY, minX, maxY, maxX]
            queue = [(y, x)]
            while queue:
                current_y, current_x = queue.pop(0)
                for (off_set_y, off_set_x) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor_y, neighbor_x = current_y + off_set_y, current_x + off_set_x
                    if 0 <= neighbor_y < height and 0 <= neighbor_x < width and np.all(image[neighbor_y][neighbor_x] != 0):
                        if colour_nearness(image[current_y][current_x], image[neighbor_y][neighbor_x], threshold) and granule_index not in granules[neighbor_y][neighbor_x]:
                            granules[neighbor_y][neighbor_x].append(granule_index)
                            queue.append((neighbor_y, neighbor_x))
                            # Aktualizowanie bounding boxa
                            bounding_boxes[granule_index][0] = min(bounding_boxes[granule_index][0], neighbor_y)  # minY
                            bounding_boxes[granule_index][1] = min(bounding_boxes[granule_index][1], neighbor_x)  # minX
                            bounding_boxes[granule_index][2] = max(bounding_boxes[granule_index][2], neighbor_y)  # maxY
                            bounding_boxes[granule_index][3] = max(bounding_boxes[granule_index][3], neighbor_x)  # maxX
            granule_index += 1
    return granules, initial_colors, bounding_boxes


def form_spatiotemporal_granules(frames, threshold, p):
    all_granules = []
    all_initial_colors = []
    all_bounding_boxes = []

    # Tworzymy granule dla każdej klatki wideo (różnica między bieżącą i poprzednią klatką)
    for i in range(1, len(frames)):
        granules, initial_colors, bounding_boxes = create_granules(frames[i], threshold)
        all_granules.append(granules)
        all_initial_colors.append(initial_colors)
        all_bounding_boxes.append(bounding_boxes)

    # Łączenie granulek z różnych klatek
    for t in range(1, len(frames) - 1):
        for granule_index in range(len(all_granules[t]) - 1):
            for previous_t in range(t - 1, t - p - 1, -1):
                # Porównanie granulek z bieżącej klatki (t) z granulkami z poprzednich klatek
                current_color = all_initial_colors[t][granule_index]
                current_bbox = all_bounding_boxes[t][granule_index]

                for prev_granule_index in range(len(all_granules[previous_t])):
                    prev_color = all_initial_colors[previous_t][prev_granule_index]
                    prev_bbox = all_bounding_boxes[previous_t][prev_granule_index]

                    # Sprawdzamy, czy granule są podobne
                    if colour_nearness(current_color, prev_color, threshold) and is_overlapping(current_bbox,
                                                                                                prev_bbox):
                        # Łączymy granule z różnych klatek
                        merge_granules(granule_index, prev_granule_index, all_granules[previous_t],
                                       all_bounding_boxes[previous_t])

    return all_granules, all_initial_colors, all_bounding_boxes


def is_overlapping(bbox1, bbox2):
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2]:
        return False
    if bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]:
        return False
    return True


def merge_granules(granule_index, prev_granule_index, granules, bounding_box):
    minY, minX, maxY, maxX = bounding_box[granule_index]
    # print(minY, minX, maxY, maxX)

    for y in range(minY, maxY + 1):
        for x in range(minX, maxX + 1):
            if prev_granule_index in granules[y][x]:
                granules[y][x] = [granule_index]


frames = video_processing.load_frames_from_mp4("../data/reka.mp4")
granules, initial_colors, bounding_boxes = form_spatiotemporal_granules(frames, 2, 3)

out = cv2.VideoWriter('../results/output_video_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frames[0].shape[1], frames[0].shape[0]))

for frame_index in range(len(frames) - 1):
    frame = frames[frame_index]
    for granule_index, bbox in bounding_boxes[frame_index].items():
        minY, minX, maxY, maxX = bbox
        cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 0), 1)

    out.write(frame)  # Dodajemy klatkę do pliku wideo

out.release()




# frame = video_processing.load_frames_from_mp4("../data/reka.mp4")[0]
# granules, initial_colors, bounding_boxes = create_granules(frame, 2) #400
#
# for granule_index, bbox in bounding_boxes.items():
#     minY, minX, maxY, maxX = bbox
#     cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 0), 1)
#
# cv2.imwrite("../results/granuled_arm.jpg", frame)

