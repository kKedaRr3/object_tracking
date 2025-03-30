import numpy as np

class Granule:
    def __init__(self, x, y, size, avg_color):
        self.x = x
        self.y = y
        self.size = size
        self.avg_col = avg_color

class Granulation:

    @staticmethod
    def granulate(self, image):
        granules = []
        for i in range(5):
            granules.append(Granule(i, i, i, i))
        return granules
        
    def quadtree_decomposition(self, image, threshold, min_size):
        frame_y, frame_x = image.shape
        intensity_difference = np.max(image) - np.min(image)
        if intensity_difference > threshold and frame_x > min_size:
            quadrants = self.split_image(image) # TODO dodac SplitImage
            for quadrant in quadrants:
                self.quadtree_decomposition(quadrant, threshold, min_size)
        else:
            center_x = frame_x // 2
            center_y = frame_y // 2
            # Store(center_x, center_y) TODO zapisac te center_x i center_y w jakiejs strukturze globalnej


    def split_image(self, image):
        frame_y, frame_x = image.shape
        offset_x = frame_x / 2
        offset_y = frame_y / 2
        first_quadrant = image[0:offset_y, 0:offset_x]
        second_quadrant = image[0:offset_y:frame_y, offset_x:frame_x]
        third_offset = image[offset_y:frame_y, 0:offset_x]
        fourth_quadrant = image[offset_y:frame_y, offset_x:frame_x]
        return first_quadrant, second_quadrant, third_offset, fourth_quadrant