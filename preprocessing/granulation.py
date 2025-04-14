import numpy as np
from preprocessing.video_preprocessing import compute_3D_difference_matrix, compute_median_matrix

def colour_nearness_rgb(color1, color2, threshold):
    '''Funkcja do sprawdzania podobiensta kolorow'''
    return np.linalg.norm(np.array(color1) - np.array(color2)) < threshold


def create_granules_color(image, threshold: int):
    '''Funkcja tworzaca granule'''

    height, width = image.shape[:2]
    granules = np.full((height, width), None)
    initial_colors = dict()
    bounding_boxes = dict()
    granule_index = 0

    for y in range(height):
        if y % 10 == 0: print(f"Processed row {y}")
        for x in range(width):
            if np.all(image[y][x]) == 0 or granules[y][x] is not None:
                continue

            # granules[y, x] = granule_index
            # initial_colors[granule_index] = image[y][x]
            # bounding_boxes[granule_index] = [y, x, y, x]  # [minY, minX, maxY, maxX]
            neighbor_found = False
            queue = [(y, x)]
            while queue:
                current_y, current_x = queue.pop(0)
                for (off_set_y, off_set_x) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    neighbor_y, neighbor_x = current_y + off_set_y, current_x + off_set_x
                    if 0 <= neighbor_y < height and 0 <= neighbor_x < width and np.all(
                            image[neighbor_y][neighbor_x]) != 0:
                        if colour_nearness_rgb(image[current_y][current_x], image[neighbor_y][neighbor_x],
                                           threshold) and granules[neighbor_y][neighbor_x] is None:
                            neighbor_found = True

                            initial_colors[granule_index] = image[y][x]
                            bounding_boxes[granule_index] = [y, x, y, x]

                            granules[neighbor_y][neighbor_x] = granule_index
                            queue.append((neighbor_y, neighbor_x))
                            bounding_boxes[granule_index][0] = min(bounding_boxes[granule_index][0], neighbor_y)  # minY
                            bounding_boxes[granule_index][1] = min(bounding_boxes[granule_index][1], neighbor_x)  # minX
                            bounding_boxes[granule_index][2] = max(bounding_boxes[granule_index][2], neighbor_y)  # maxY
                            bounding_boxes[granule_index][3] = max(bounding_boxes[granule_index][3], neighbor_x)  # maxX

            if neighbor_found:
                granule_index += 1
    print("Frame Processed")
    return (granules, initial_colors, bounding_boxes)

def form_spatiotemporal_granules(frames, threshold):

    prev_taus = []
    granules_last, initial_colors_last, bounding_boxes_last = create_granules_color(frames[-1], threshold)

    for i in range(0, len(frames) - 1):
        granules, initial_colors, bounding_boxes = create_granules_color(frames[i], threshold)
        prev_taus.append((granules, initial_colors, bounding_boxes))

    max_last = find_max_granule_index(granules_last)
    for granule_index in range(max_last):
        if granule_index % 10 == 0: print(f"Przetwarzanie granuli {granule_index}/{max_last}")
        recurrent_function(granule_index, granules_last, initial_colors_last[granule_index],
                           bounding_boxes_last[granule_index], prev_taus, threshold)

    return prev_taus[0][0], prev_taus[0][1], prev_taus[0][2]

def recurrent_function(granule_index, current_tau_granules, current_granule_colour, current_granule_bbox, prev_taus,
                       threshold):
    if len(prev_taus) in (0, 1):
        return
    prev_tau = prev_taus[-1]
    max_prev = find_max_granule_index(prev_tau[0])
    if granule_index >= max_prev:
        return

    for prev_granule_index in range(max_prev):
        is_overlap = is_overlapping(current_granule_bbox, prev_tau[2][prev_granule_index])
        if colour_nearness_rgb(current_granule_colour, prev_tau[1][prev_granule_index], threshold) and is_overlap:
            prev_tau[1][prev_granule_index] = current_granule_colour
            merge_granules(granule_index, prev_granule_index, prev_tau[0], prev_tau[2][prev_granule_index])
            prev_colour = prev_tau[1][prev_granule_index]
            prev_bounding_box = prev_tau[2][prev_granule_index]
            recurrent_function(granule_index, prev_tau[0], prev_colour, prev_bounding_box, prev_taus[:-1], threshold)


def is_overlapping(bbox1, bbox2):
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2]:
        return False
    if bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]:
        return False
    return True


def merge_granules(granule_index, prev_granule_index, granules, bounding_box):
    minY, minX, maxY, maxX = bounding_box
    for y in range(minY, maxY + 1):
        for x in range(minX, maxX + 1):
            if prev_granule_index == granules[y][x]:
                granules[y][x] = granule_index

def find_max_granule_index(granules):
    max = 0
    for y in range(len(granules)):
        for x in range(len(granules[y])):
            if granules[y][x] is None: continue
            if granules[y][x] > max:
                max = granules[y][x]
    return max

# TODO do sprawdzenia bo moze byc zle
def form_rgb_d_granules(sp_t_granules, sp_t_initial_colors, sp_t_bounding_boxes, threshold):
    sp_t_image = np.zeros_like(sp_t_granules)
    for y in range(sp_t_image.shape[0]):
        for x in range(sp_t_image.shape[1]):
            if sp_t_granules[y, x] is not None:
                sp_t_image[y, x] = sp_t_initial_colors[sp_t_granules[y, x]]

    granules, initial_colors, bounding_boxes = create_granules_color(sp_t_image, threshold)

    return granules, initial_colors, bounding_boxes
