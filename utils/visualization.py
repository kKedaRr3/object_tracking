import networkx as nx
from matplotlib import pyplot as plt

import cv2
from preprocessing.granulation import *
from preprocessing.video_preprocessing import compute_3D_difference_matrix, compute_median_matrix


class Visualization:

    @staticmethod
    def visualize_spatiotemporal_granules(frames, output_path, bbox=True, bbox_color=(255, 0, 0)):
        if len(frames) <= 3:
            raise Exception('Not enough frames to visualize')
        diff_frames = compute_3D_difference_matrix(frames[:-1], frames[-1])
        median_frame = compute_median_matrix(diff_frames)
        threshold = 0.3 * np.max(median_frame)

        granules, initial_colors, bounding_boxes = form_spatiotemporal_granules(diff_frames, threshold)

        if bbox:
            for granule_index, bbox in bounding_boxes.items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(median_frame, (minX, minY), (maxX, maxY), bbox_color, 1)
            cv2.imwrite(output_path, median_frame)
        else:
            result = np.zeros_like(frames[0])
            for y in range(frames[0].shape[0]):
                for x in range(frames[0].shape[1]):
                    if granules[y, x] != -1:
                        result[y, x] = initial_colors[granules[y, x]]
            cv2.imwrite(output_path, result)

    @staticmethod
    def visualize_spatio_color_granules(image, output_path, threshold=2, bbox=True, bbox_color=(0, 0, 0)):
        if type(image).__name__ == 'str':
            image = cv2.imread(image)

        granules, initial_colors, bounding_boxes = create_granules_color(image, threshold)

        if bbox:
            for granule_index, bbox in bounding_boxes.items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(image, (minX, minY), (maxX, maxY), bbox_color, 1)
            cv2.imwrite(output_path, image)

        else:
            result = np.zeros_like(image)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if granules[y, x] != -1:
                        result[y, x] = initial_colors[granules[y, x]]
            cv2.imwrite(output_path, result)

    @staticmethod
    def visualize_rgb_granules(frames, output_path, bbox=True, bbox_color=(255, 0, 0)):
        if len(frames) <= 3:
            raise Exception('Not enough frames to visualize')
        diff_frames = compute_3D_difference_matrix(frames[:-1], frames[-1])
        median_frame = compute_median_matrix(diff_frames)
        threshold = 0.3 * np.max(median_frame)

        granules, initial_colors, bounding_boxes = form_spatiotemporal_granules(diff_frames, threshold)
        rgb_granules, rgb_initial_colors, rgb_bounding_boxes = form_rgb_d_granules(granules, initial_colors,
                                                                                   bounding_boxes, threshold)

        if bbox:
            for granule_index, bbox in rgb_bounding_boxes.items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(frames[0], (minX, minY), (maxX, maxY), bbox_color, 1)
            cv2.imwrite(output_path, frames[0])
        else:
            result = np.zeros_like(frames[0])
            for y in range(frames[0].shape[0]):
                for x in range(frames[0].shape[1]):
                    if granules[y, x] != -1:
                        result[y, x] = rgb_initial_colors[rgb_granules[y, x]]
            cv2.imwrite(output_path, result)

    @staticmethod
    def visualize_d_granules(frames, output_path, bbox=True, bbox_color=(255, 0, 0)):
        if len(frames) <= 3:
            raise Exception('Not enough frames to visualize')

        depth_diff_3D_matrix = compute_3D_difference_matrix(frames[:-1], frames[-1])
        depth_median_matrix = compute_median_matrix(depth_diff_3D_matrix)
        cv2.imwrite("../results/man/diff_median.jpg", depth_median_matrix)
        threshold = 0.3 * np.max(depth_median_matrix)
        sp_t_gib = form_spatiotemporal_granules(depth_diff_3D_matrix, threshold)
        d_gib = form_rgb_d_granules(sp_t_gib[0], sp_t_gib[1], sp_t_gib[2], 15)

        if bbox:
            for granule_index, bbox in d_gib[2].items():
                minY, minX, maxY, maxX = bbox
                cv2.rectangle(frames[0], (minX, minY), (maxX, maxY), bbox_color, 1)
            cv2.imwrite(output_path, frames[0])
        else:
            result = np.zeros_like(frames[0])
            for y in range(frames[0].shape[0]):
                for x in range(frames[0].shape[1]):
                    if d_gib[0][y, x] != -1:
                        result[y, x] = d_gib[1][d_gib[0][y, x]]
            cv2.imwrite(output_path, result)


    @staticmethod
    def draw_flow_graph(graph):
        pos = {
            "O": (0, 2),
            "B": (0, -2),

            "NBT": (4, 6),
            "CCT": (4, 2),
            "PBT": (4, -2),
            "BeT": (4, -6),

            "NBR": (8, 6),
            "PBR": (8, -2),
            "BeR": (8, -6),

            "NBD": (12, 6),
            "CCD": (12, 2),
            "PBD": (12, -2),
            "BeD": (12, -6),

            "D1": (16, 2),
            "D2": (16, -6)
        }
        plt.figure(figsize=(12, 8))

        node_weights = nx.get_node_attributes(graph, "weight")

        node_labels = {node: f"{node}\n{weight}" for node, weight in node_weights.items()}

        edge_labels = nx.get_edge_attributes(graph, 'weight')

        nx.draw(graph, pos, with_labels=False)

        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, label_pos=0.25)

        nx.draw_networkx_labels(graph, pos, labels=node_labels)

        plt.show()
