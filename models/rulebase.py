from preprocessing.granulation import find_max_granule_index
import numpy as np


def initialize_rule_base(spatio_temporal_granules, rgb_d_granules, background):

    '''
    Funkcja inicjalizujaca baze regul ale czy to ma tak byc to nie wiem
    trzeba to zrobic na pdostawie strony 39 z magisterki
    jesli wszsystkie piksele z spt/rgb-d granuli znajduja sie w obiekcie(obiekt wiekszy od granuli) to jest BE
    jesli czesc pikseli z spt/rgb-d granuli znajduja sie w obiekcie to jest PB
    jesli zadne piksele z spt/rgb-d lub za malo piskeli znajduja sie w obiekcie to jest NB
    jesli wszystkie piksele z obiektu znajduja sie w spt/rgb-d granuli(granula wieksza niz obiekt) to jest CC
    :param spatio_temporal_granules: granule [0] czasowo przestrzenne z ich granicami [2] i kolorami [1]
    :param rgb_d_granules: granule rgb i glebokosci
    :param background: binarny obraz tla klatki otrzymany przez three-point-approximation
    :return: poczatkowa baza regul
    '''

    rule_base = {}

    spt_max_index = find_max_granule_index(spatio_temporal_granules[0])
    rgb_d_max_index = find_max_granule_index(rgb_d_granules[0])

    # Dolne przybliżenie obiektu (O)
    O = []  # Zbiór dla obiektów
    for granule_index in range(spt_max_index):
        if is_object(spatio_temporal_granules[2][granule_index], background):
            O.append(granule_index)

    # Górne przybliżenie obiektu (O)
    O_upper = {}  # Zbiór dla obiektów - górne przybliżenie
    for granule_index, granule in enumerate(spatio_temporal_granules[0]):
        # Jeśli granula jest częścią obiektu lub zmienia się w czasie
        if is_object(granule) or is_similar_to_previous(granule, spatio_temporal_granules[0]):
            O_upper[granule_index] = granule

    # Cechy temporalne (Frame difference in RGB-D feature space)
    temporal_features = {}
    for granule_index, granule in enumerate(spatio_temporal_granules[0]):
        temporal_features[granule_index] = compute_temporal_features(granule)

    # Cechy koloru (RGB-D values)
    color_features = {}
    for granule_index, granule in enumerate(rgb_d_granules[0]):
        color_features[granule_index] = compute_color_features(granule)

    # Cechy przestrzenne (Spatial location)
    spatial_features = {}
    for granule_index, granule in enumerate(spatio_temporal_granules[0]):
        spatial_features[granule_index] = compute_spatial_location(granule)

    # Reguły dla bazy reguł
    for granule_index in O:
        rule_base[granule_index] = {
            "temporal": temporal_features[granule_index],
            "color": color_features[granule_index],
            "spatial": spatial_features[granule_index],
            "type": "object"  # Przypisanie granuli jako obiekt
        }

    for granule_index in O_upper.keys():
        rule_base[granule_index] = {
            "temporal": temporal_features[granule_index],
            "color": color_features[granule_index],
            "spatial": spatial_features[granule_index],
            "type": "object_upper"  # Przypisanie granuli jako górne przybliżenie obiektu
        }

    # Dodanie tła do bazy reguł
    for granule_index, granule in enumerate(background):
        rule_base[granule_index] = {
            "temporal": temporal_features.get(granule_index, None),
            "color": color_features.get(granule_index, None),
            "spatial": spatial_features.get(granule_index, None),
            "type": "background"  # Przypisanie granuli jako tło
        }

    return rule_base


# funkcja do sprawdzania czy granula jest obiektem (wszystkie jej wartosci w granicach bounding box sa rowne 0 [0 - obiekt, 255 - tlo])
def is_object(bbox, background):
    minY, minX, maxY, maxX = bbox
    return np.all(background[minY:maxY+1, minX:maxX+1]) == 0

def is_similar_to_previous(granule, granules):
    # Funkcja do sprawdzenia, czy granula jest podobna do poprzednich
    return True  # Należy wprowadzić odpowiednią logikę porównania

def compute_temporal_features(granule):
    # Funkcja do obliczania cech temporalnych (różnice między klatkami)
    return {"temporal_feature": 0}  # Zwraca przykładowe cechy

def compute_color_features(granule):
    # Funkcja do obliczania cech koloru
    return {"color_feature": 0}  # Zwraca przykładowe cechy

def compute_spatial_location(granule):
    # Funkcja do obliczania cech przestrzennych
    return {"spatial_feature": (0, 0)}  # Zwraca przykładowe cechy
