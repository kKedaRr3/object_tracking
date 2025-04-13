from preprocessing.granulation import find_max_granule_index
import numpy as np

def initialize_rule_base(spatio_color_gib, spatio_temporal_gib, rgb_gib, d_gib):

    '''
    Trzeba porownac spatio_color granules(otrzymane z create_granules) z spatio-temporal granules(otrzymane z form_spatiotemporal_granules) rgb_granules(otrzymane z form_rgb_d_granules) i d_granules

    :param spatio_color_gib:
    :param spatio_temporal_gib:
    :param rgb_gib:
    :param d_gib:
    :return: poczatkowa baza regul gdzie 2 to objekt, 1 to tlo a 0 to niezidentyfikowany obiekt
    '''

    height, width = rgb_gib[0].shape
    spatiotemporal_features = np.zeros_like(spatio_color_gib[0])
    rgb_features = np.zeros_like(spatio_color_gib[0])
    d_features = np.zeros_like(spatio_color_gib[0])

    for y in range(height):
        for x in range(width):
            label = spatio_color_gib[0][y][x]
            # Maska pojedynczej granuli spatio_color
            spatio_color_granule = spatio_temporal_gib[0][y][x] == label
            spatiotemporal_features = calculate_attribute(spatio_color_granule, spatio_temporal_gib[0], x, y)
            rgb_features = calculate_attribute(spatio_color_granule, rgb_gib[0], x, y)
            d_features = calculate_attribute(spatio_color_granule, d_gib[0], x, y)

    result_object = np.logical_or(
        np.logical_and.reduce((spatiotemporal_features == 1, rgb_features == 2), (d_features == 2)),
        np.logical_and.reduce((spatiotemporal_features == 2, rgb_features == 2), (d_features == 1)),
        np.logical_and.reduce((spatiotemporal_features == 3, rgb_features == 2, d_features == 2)),
        np.logical_and.reduce((spatiotemporal_features == 2, rgb_features == 0, d_features == 0)),
        np.logical_and.reduce((spatiotemporal_features == 1, rgb_features == 1, d_features == 3)),
        np.logical_and.reduce((spatiotemporal_features == 2, rgb_features == 2, d_features == 2)),
    )

    result_background = np.logical_or.reduce(
        np.logical_and.reduce((spatiotemporal_features == 0, rgb_features == 0, d_features == 0)),
        np.logical_and.reduce((spatiotemporal_features == 0, rgb_features == 2, d_features == 2)),
        np.logical_and.reduce((spatiotemporal_features == 1, rgb_features == 2, d_features == 1)),
        np.logical_and.reduce((spatiotemporal_features == 0, rgb_features == 0, d_features == 2)),
        np.logical_and.reduce((spatiotemporal_features == 2, rgb_features == 0, d_features == 0)),
    )

    rule_base = np.zeros_like(result_object, np.uint8)
    rule_base[result_object] = 2
    rule_base[result_background] = 1

    return rule_base


def calculate_attribute(spatio_color_granule, granules_to_calculate, y, x):
    label = granules_to_calculate[y][x]
    granule = granules_to_calculate == label
    intersection = np.logical_and(spatio_color_granule, granule)
    return get_attribute(spatio_color_granule, granule, intersection)


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
        raise ValueError("Somehow, the intersection doesn't match the intersection of the two granule types.")

