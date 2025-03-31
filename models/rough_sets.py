import numpy as np

def lower_approximation(U, X, ind_relation):
    """
    U  - pełna dziedzina (lista lub zbiór obiektów)
    X  - zbiór obiektów, który przybliżamy
    ind_relation - relacja nieodróżnialności (lista par lub słownik)
    Zwraca obiekty, które na pewno należą do X
    """
    lower_approx = []
    for u in U:
        # klasa ekwiwalencji dla obiektu u
        eq_class = ind_relation[u]
        # jeśli cała klasa ekwiwalencji mieści się w X
        if eq_class.issubset(X):
            lower_approx.append(u)
    return set(lower_approx)

def upper_approximation(U, X, ind_relation):
    """
    Zwraca obiekty, które mogą należeć do X
    """
    upper_approx = []
    for u in U:
        eq_class = ind_relation[u]
        # jeśli klasa ekwiwalencji ma część wspólną z X
        if eq_class.intersection(X):
            upper_approx.append(u)
    return set(upper_approx)

def roughness(U, X, ind_relation):
    """
    Wzór: R_B(X) = 1 - |B¯(X)| / |B̄(X)|
    O ile B¯(X) i B̄(X) != pusty
    """
    la = lower_approximation(U, X, ind_relation)
    ua = upper_approximation(U, X, ind_relation)
    if len(ua) == 0:
        return 0.0
    return 1.0 - (len(la) / len(ua))

def rough_entropy(object_roughness, background_roughness, base=10):
    """
    Wzór na rough entropy:
    RE_T = - (base/2) * [ R_obj * log_base(R_obj) + R_bg * log_base(R_bg) ]
    z uwzględnieniem warunków brzegowych.
    """
    # Zgodnie z ideą z pracy - jeśli roughness < 1/base, to liczymy inaczej
    def entropy_part(r_val):
        if r_val <= 1.0/base:
            return 1.0
        else:
            return r_val * np.log(r_val) / np.log(base)

    e_obj = entropy_part(object_roughness)
    e_bg  = entropy_part(background_roughness)
    re_val = - (base / 2.0) * (e_obj + e_bg)
    return re_val
