import numpy as np
from models.rough_sets import rough_entropy, roughness


def compute_spatial_segmentation(gray_image, granules, base=10):
    """
    Szukanie optymalnego progu T* metodą rough entropy
    Na bazie granulek (np. quadtree albo region-growing).
    Zwraca binary_mask - maskę (0/1) dla tła/obiektu.
    """
    min_g = np.min(gray_image)
    max_g = np.max(gray_image)

    best_thresh = min_g
    best_re     = -9999

    for T in range(min_g, max_g+1):
        # Oblicz roughness obiektu i tła w oparciu o granule i T
        # (Tutaj tylko przykład uproszczony)
        object_rough = 0.0
        background_rough = 0.0

        # Pomysł: zlicz granule, które w całości > T albo < T, itp.
        # Potem oblicz R_obj, R_bg i entropię
        # Poniżej symbolicznie:
        R_obj = 0.5  # tu musiałbyś wstawić właściwe obliczenia
        R_bg  = 0.5

        re_val = rough_entropy(R_obj, R_bg, base=base)
        if re_val > best_re:
            best_re = re_val
            best_thresh = T

    # Zwróć binary maskęthree_point_approximation
    bin_mask = (gray_image >= best_thresh).astype(np.uint8)
    return bin_mask, best_thresh
