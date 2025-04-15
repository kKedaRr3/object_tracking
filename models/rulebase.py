from preprocessing.granulation import find_max_granule_index
import numpy as np
from numba import njit

def generate_rule_base(spatio_color_gib, spatio_temporal_gib, rgb_gib, d_gib):

    '''
    Trzeba porownac spatio_color granules(otrzymane z create_granules) z spatio-temporal granules(otrzymane z form_spatiotemporal_granules) rgb_granules(otrzymane z form_rgb_d_granules) i d_granules

    :param spatio_color_gib:
    :param spatio_temporal_gib:
    :param rgb_gib:
    :param d_gib:
    :return: poczatkowa baza regul gdzie 2 to objekt, 1 to tlo a 0 to niezidentyfikowany
    '''

    height, width = spatio_color_gib[0].shape

    spatiotemporal_features = dict()
    rgb_features = dict()
    d_features = dict()
    result_object = dict()
    result_background = dict()
    rule_base = np.zeros_like(spatio_color_gib[0])

    max_index = find_max_granule_index(spatio_color_gib[0])
    for label in range(max_index + 1):
        if label % 50 == 0: print(f"Processing granule {label}/{max_index}")
        spatio_color_granule = spatio_color_gib[0] == label
        # biore srodek granuli ale czy to jest poprawnie to nie wiem a juz tym bardziej czy optymalne
        minY, minX, maxY, maxX  = spatio_color_gib[2][label]
        y, x = int((maxY + minY) / 2), int((maxX + minX) / 2)
        spatiotemporal_features[label] = calculate_attribute(spatio_color_granule, spatio_temporal_gib[0], y, x, spatio_color_gib[2][label])
        rgb_features[label] = calculate_attribute(spatio_color_granule, rgb_gib[0], y, x, spatio_color_gib[2][label])
        d_features[label] = calculate_attribute(spatio_color_granule, d_gib[0], y, x, spatio_color_gib[2][label])

        result_object[label] = np.logical_or.reduce((
            np.logical_and.reduce((spatiotemporal_features[label] == 1, rgb_features[label] == 2, d_features[label] == 2)),
            np.logical_and.reduce((spatiotemporal_features[label] == 2, rgb_features[label] == 2, d_features[label] == 1)),
            np.logical_and.reduce((spatiotemporal_features[label] == 3, rgb_features[label] == 2, d_features[label] == 2)),
            np.logical_and.reduce((spatiotemporal_features[label] == 2, rgb_features[label] == 0, d_features[label] == 0)),
            np.logical_and.reduce((spatiotemporal_features[label] == 1, rgb_features[label] == 1, d_features[label] == 3)),
            np.logical_and.reduce((spatiotemporal_features[label] == 2, rgb_features[label] == 2, d_features[label] == 2)),
        ))

        result_background[label] = np.logical_or.reduce((
            np.logical_and.reduce((spatiotemporal_features[label] == 0, rgb_features[label] == 0, d_features[label] == 0)),
            np.logical_and.reduce((spatiotemporal_features[label] == 0, rgb_features[label] == 2, d_features[label] == 2)),
            np.logical_and.reduce((spatiotemporal_features[label] == 1, rgb_features[label] == 2, d_features[label] == 1)),
            np.logical_and.reduce((spatiotemporal_features[label] == 0, rgb_features[label] == 0, d_features[label] == 2)),
            np.logical_and.reduce((spatiotemporal_features[label] == 2, rgb_features[label] == 0, d_features[label] == 0)),
        ))


    for y in range(height):
        for x in range(width):
            label = spatio_color_gib[0][y][x]
            if label is None: continue
            if result_object[label] == 1:
                rule_base[y][x] = 2
            elif result_background[label] == 1:
                rule_base[y][x] = 1
            else:
                rule_base[y][x] = 0

    features = (spatiotemporal_features, rgb_features, d_features, result_object, result_background)

    return rule_base, features


def calculate_attribute(spatio_color_granule, granules_to_calculate, y, x, bbox):
    label = granules_to_calculate[y][x]

    minY, minX, maxY, maxX = bbox
    test = []
    for y_t in range(minY, maxY):
        for x_t in range(minX, maxX):
            label_test = granules_to_calculate[y_t][x_t]
            if spatio_color_granule[y_t][x_t] == 0: continue
            if label_test not in test and label_test is not None:
                test.append(label_test)
    if len(test) > 1:
        print("\n\njest wiecej granul bazowych niz 1 wewnatrz granuli spatio_color")
        print(test)

    # label = granules_to_calculate[y][x]
    granule = granules_to_calculate == label
    intersection = np.logical_and(spatio_color_granule, granule)
    return get_attribute(spatio_color_granule, granule, intersection)


@njit()
def get_attribute(spatio_color_granule, granule, intersection):
    '''
    :param spatio_color_granule:
    :param granule:
    :param intersection:
    :return: 0 to NB,   1 to PB,    2 to Be,    3, to CC
    '''

    sum_intersection = np.sum(intersection)
    sum_spatio_color_granule = np.sum(spatio_color_granule)
    sum_granule = np.sum(granule)

    if sum_intersection == 0:
        return 0
    elif sum_spatio_color_granule >= sum_granule == sum_intersection:
        return 3
    elif sum_intersection == sum_spatio_color_granule < sum_granule:
        return 2
    elif sum_intersection < sum_spatio_color_granule:
        return 1
    else:
        print(sum_intersection, sum_spatio_color_granule, sum_granule)
        print(intersection.dtype)
        print(spatio_color_granule.dtype)
        print(granule.dtype)
        raise ValueError("Somehow, the intersection doesn't match the intersection of the two granule types.")
