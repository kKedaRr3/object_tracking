import numpy as np

'''Funkcja do sprawdzania podobiensta kolorow'''


def colour_nearness(color1, color2, threshold):
    distance = np.linalg.norm(np.array(color1) - np.array(color2))
    return distance < threshold


'''Funkcja tworzaca granule  wersja chyba bez nakladajacych sie na siebie granuli'''


def create_granules(image, threshold):
    height, width = image.shape[:2]
    granules = [[None for _ in range(width)] for _ in range(height)]
    initial_colors = dict()
    bounding_boxes = dict()
    granule_index = 0

    for y in range(height):
        if y % 10 == 0: print(f"Processed row {y}")
        for x in range(width):
            if np.all(image[y][x]) == 0 or granules[y][x] is not None:
                continue
            initial_colors[granule_index] = image[y][x]
            bounding_boxes[granule_index] = [y, x, y, x]  # [minY, minX, maxY, maxX]
            queue = [(y, x)]
            while queue:
                current_y, current_x = queue.pop(0)
                for (off_set_y, off_set_x) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor_y, neighbor_x = current_y + off_set_y, current_x + off_set_x
                    if 0 <= neighbor_y < height and 0 <= neighbor_x < width and np.all(
                            image[neighbor_y][neighbor_x]) != 0:
                        if colour_nearness(image[current_y][current_x], image[neighbor_y][neighbor_x],
                                           threshold) and granules[neighbor_y][neighbor_x] is None:
                            granules[neighbor_y][neighbor_x] = granule_index
                            queue.append((neighbor_y, neighbor_x))
                            bounding_boxes[granule_index][0] = min(bounding_boxes[granule_index][0], neighbor_y)  # minY
                            bounding_boxes[granule_index][1] = min(bounding_boxes[granule_index][1], neighbor_x)  # minX
                            bounding_boxes[granule_index][2] = max(bounding_boxes[granule_index][2], neighbor_y)  # maxY
                            bounding_boxes[granule_index][3] = max(bounding_boxes[granule_index][3], neighbor_x)  # maxX
            granule_index += 1
    print("Frame Processed")
    return granules, initial_colors, bounding_boxes


'''
p - Odnosi się do liczby klatek, które są uwzględniane w analizie różnic pomiędzy klatkami. 
Oznacza to, że algorytm bierze pod uwagę p poprzednich klatek (w tym przypadku różnicę między bieżącą klatką a poprzednimi p klatkami) 
i na tej podstawie tworzy granule, które będą analizowane pod kątem zmian w czasie.
'''


def form_spatiotemporal_granules(frames, threshold, p):
    prev_taus = []
    # granule dla ostatniej klatki
    granules_last, initial_colors_last, bounding_boxes_last = create_granules(frames[-1], threshold)

    for i in range(0, len(frames) - 1):
        granules, initial_colors, bounding_boxes = create_granules(frames[i], threshold)
        prev_taus.append((granules, initial_colors, bounding_boxes))

    max = find_max_granule_index(granules_last)
    for granule_index in range(max):
        if granule_index % 10 == 0: print(f"Przetwarzanie granuli {granule_index}/{max}")
        recurrent_function(granule_index, granules_last, initial_colors_last[granule_index],
                           bounding_boxes_last[granule_index], prev_taus, threshold)

    prev_taus.append((granules_last, initial_colors_last, bounding_boxes_last))

    return prev_taus


def recurrent_function(granule_index, current_tau_granules, current_granule_colour, current_granule_bbox, prev_taus,
                       threshold):
    if len(prev_taus) in (0, 1):
        return
    prev_tau = prev_taus[-1]
    max_prev = find_max_granule_index(prev_tau[0])
    if granule_index >= max_prev:
        return

    for prev_granule_index in range(find_max_granule_index(prev_tau[0])):
        is_overlap = is_overlapping(current_granule_bbox, prev_tau[2][prev_granule_index])
        if colour_nearness(current_granule_colour, prev_tau[1][prev_granule_index], threshold) and is_overlap:
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
