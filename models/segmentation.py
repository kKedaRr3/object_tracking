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

    # Zwróć binary maskę
    bin_mask = (gray_image >= best_thresh).astype(np.uint8)
    return bin_mask, best_thresh

def three_point_approximation(prev_frames, current_frame, e=3.0):
    """
    prev_frames: lista np. 3 poprzednich klatek
    e=3.0 – zwykle 3 * odchylenie standardowe
    Zwraca maskę obiektu (1=obiekt, 0=tło) na podstawie min, median i max.
    """
    # Załóżmy, że prev_frames ma dokładnie 3 obrazy
    f1, f2, f3 = prev_frames
    # Wyliczamy optimistic, median, pessimistic:
    optimistic = np.max(np.stack([f1, f2, f3]), axis=0)
    pessimistic = np.min(np.stack([f1, f2, f3]), axis=0)
    median_val = np.median(np.stack([f1, f2, f3]), axis=0)

    # Mean i std (3-point approximation)
    mean_est = (optimistic + 4*median_val + pessimistic) / 6.0
    std_est  = (optimistic - pessimistic) / 6.0  # uproszczona std

    diff = np.abs(current_frame - mean_est)
    mask = (diff > e * std_est).astype(np.uint8)  # 1 jeśli obiekt, 0 – tło

    return mask
