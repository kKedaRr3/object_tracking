import numpy as np

class QuadGranule:
    def __init__(self, x, y, w, h, avg_val):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.avg_val = avg_val

def quadtree_decomposition(gray_image, threshold=10, min_size=8,
                           start_x=0, start_y=0, granules=None):
    """
    Rekurencyjny podział gray_image na kwadranty.
    Zwraca listę granulek (QuadGranule).
    """
    if granules is None:
        granules = []

    h, w = gray_image.shape[:2]
    region = gray_image[start_y:start_y+h, start_x:start_x+w]
    diff_val = np.max(region) - np.min(region)

    if diff_val < threshold or h <= min_size or w <= min_size:
        avg_val = np.mean(region)
        granules.append(QuadGranule(start_x, start_y, w, h, avg_val))
    else:
        half_h = h // 2
        half_w = w // 2
        # Lewy górny
        quadtree_decomposition(
            gray_image[start_y:start_y+half_h, start_x:start_x+half_w],
            threshold, min_size, 0, 0, granules
        )
        # Prawy górny
        quadtree_decomposition(
            gray_image[start_y:start_y+half_h, start_x+half_w:start_x+w],
            threshold, min_size, 0, 0, granules
        )
        # Lewy dolny
        quadtree_decomposition(
            gray_image[start_y+half_h:start_y+h, start_x:start_x+half_w],
            threshold, min_size, 0, 0, granules
        )
        # Prawy dolny
        quadtree_decomposition(
            gray_image[start_y+half_h:start_y+h, start_x+half_w:start_x+w],
            threshold, min_size, 0, 0, granules
        )

    return granules
