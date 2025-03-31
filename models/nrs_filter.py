import numpy as np

def nrs_filter(deltas_list, threshold1, threshold2):
    """
    deltas_list - lista różnic pomiędzy kolejnymi klatkami
    Zwraca np. (lower_approx, upper_approx), uproszczony przykład
    """
    # W pracy jest opis: δi to zbiory granulek zmienionych między f_i a f_{i-1}
    # δP to zbiory zmienione między f_t i f_{t - P}, itp.
    # Tu symbolicznie:
    union_set  = set()  # Uc
    intersect_set = None  # Ic

    for dset in deltas_list:
        if intersect_set is None:
            intersect_set = dset
        else:
            intersect_set = intersect_set.intersection(dset)
        union_set = union_set.union(dset)

    lower_approx = intersect_set  # O¯C
    upper_approx = union_set      # ŌC

    return lower_approx, upper_approx

def compute_snr(lower_set, boundary_set):
    """
    SNR = 20 * log10( |lower| / (|boundary| - |lower| ) )
    boundary = upper - lower
    """
    L = len(lower_set)
    B = len(boundary_set)
    if (B - L) == 0:
        return 999.0
    import math
    val = L / (B - L)
    snr_db = 20 * math.log10(val) if val>0 else -999.0
    return snr_db
