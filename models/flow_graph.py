import networkx as nx
import numpy as np

from utils.visualization import Visualization

#TODO
# Do przeanalizowania/zrobienia:
# 1) W rule base jest duzo niezakwalifikowanych granul ktore nie sa ani obiektem ani tlem przez co wydaje mi sie ze sa takie dziwne przeplywy
# 2) Zrobic porownanie dwoch grafow

# a:b to jest rozklad klas wejsciowych to znaczy jaka jest proporcja miedzy granulami obiekow a granulami tla
# alfa:beta to proporcja miedzy klasami w nowym rule_base

def generate_flow_graph(features):
    flow_graph = generate_empty_graph()
    add_weights_to_flow_graph(flow_graph, features)
    return flow_graph

def compute_rule_base_coverage(g1, features):

    nfg1 = compute_normalized_graph(g1)
    nfg1 = compute_expected_graph(nfg1)

    g2 = generate_flow_graph(features)
    nfg2 = compute_normalized_graph(g2)

    return compute_mutual_dependency(nfg1, nfg2)

def compute_normalized_graph(fg2):
    all_granules_count = fg2["O"] + fg2["B"]
    for _, _, data in fg2.edges(data=True):
        data["weight"] = data["weight"] / all_granules_count
    for _, data in fg2.nodes(data=True):
        data["weight"] = data["weight"] / all_granules_count
    return fg2

def compute_expected_graph(nfg):
    pass

def compute_mutual_dependency(fg1, fg2):
    sum_top, sum_bottom = 0, 0
    for u, v, data in fg1.edges(data=True):
        sum_top += np.abs(data["weight"] - fg2[u][v]["weight"])
        sum_bottom += data["weight"]
    return sum_top / sum_bottom

def generate_empty_graph():
    graph = nx.DiGraph()

    graph.add_nodes_from(["O", "B"])
    graph.add_nodes_from(["NBT", "CCT", "PBT", "BeT"])
    graph.add_nodes_from(["NBR", "PBR", "BeR"])
    graph.add_nodes_from(["NBD", "CCD", "PBD", "BeD"])
    graph.add_nodes_from(["D1", "D2"])

    graph.add_edges_from([("O", "NBT"), ("O", "CCT"), ("O", "PBT"), ("O", "BeT")], weight=0)
    graph.add_edges_from([("B", "NBT"), ("B", "PBT"), ("B", "BeT")], weight=0)

    graph.add_edges_from([("NBT", "NBR"), ("NBT", "BeR")], weight=0)
    graph.add_edges_from([("CCT", "BeR")], weight=0)
    graph.add_edges_from([("PBT", "PBR"), ("PBT", "BeR")], weight=0)
    graph.add_edges_from([("BeT", "NBR"), ("BeT", "BeR")], weight=0)

    graph.add_edges_from([("NBR", "NBD"), ("NBR", "BeD")], weight=0)
    graph.add_edges_from([("PBR", "CCD"), ("PBR", "PBD")], weight=0)
    graph.add_edges_from([("BeR", "PBD"), ("BeR", "BeD")], weight=0)

    graph.add_edges_from([("NBD", "D1"), ("NBD", "D2")], weight=0)
    graph.add_edges_from([("CCD", "D1")], weight=0)
    graph.add_edges_from([("PBD", "D1")], weight=0)
    graph.add_edges_from([("BeD", "D1")], weight=0)

    return graph


def add_weights_to_flow_graph(graph, features):
    sp_t_features, rgb_features, d_features, object_granules, background_granules = features
    for granule_index in object_granules.keys():

        if object_granules[granule_index] == 0 and background_granules[granule_index] == 0: continue

        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 0:
            graph["O"]["NBT"]["weight"] += 1
        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 3:
            graph["O"]["CCT"]["weight"] += 1
        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 2:
            graph["O"]["BeT"]["weight"] += 1
        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 1:
            graph["O"]["PBT"]["weight"] += 1

        if background_granules[granule_index] == 1 and sp_t_features[granule_index] == 0:
            graph["B"]["NBT"]["weight"] += 1
        if background_granules[granule_index] == 1 and sp_t_features[granule_index] == 1:
            graph["B"]["PBT"]["weight"] += 1
        if background_granules[granule_index] == 1 and sp_t_features[granule_index] == 2:
            graph["B"]["BeT"]["weight"] += 1


        if sp_t_features[granule_index] == 0 and rgb_features[granule_index] == 0:
            graph["NBT"]["NBR"]["weight"] += 1
        if sp_t_features[granule_index] == 0 and rgb_features[granule_index] == 2:
            graph["NBT"]["BeR"]["weight"] += 1

        if sp_t_features[granule_index] == 3 and rgb_features[granule_index] == 2:
            graph["CCT"]["BeR"]["weight"] += 1

        if sp_t_features[granule_index] == 1 and rgb_features[granule_index] == 1:
            graph["PBT"]["PBR"]["weight"] += 1
        if sp_t_features[granule_index] == 1 and rgb_features[granule_index] == 2:
            graph["PBT"]["BeR"]["weight"] += 1

        if sp_t_features[granule_index] == 2 and rgb_features[granule_index] == 0:
            graph["BeT"]["NBR"]["weight"] += 1
        if sp_t_features[granule_index] == 2 and rgb_features[granule_index] == 2:
            graph["BeT"]["BeR"]["weight"] += 1


        if rgb_features[granule_index] == 0 and d_features[granule_index] == 0:
            graph["NBR"]["NBD"]["weight"] += 1
        if rgb_features[granule_index] == 0 and d_features[granule_index] == 2:
            graph["NBR"]["BeD"]["weight"] += 1

        if rgb_features[granule_index] == 1 and d_features[granule_index] == 3:
            graph["PBR"]["CCD"]["weight"] += 1
        if rgb_features[granule_index] == 1 and d_features[granule_index] == 1:
            graph["PBR"]["PBD"]["weight"] += 1

        if rgb_features[granule_index] == 2 and d_features[granule_index] == 1:
            graph["BeR"]["PBD"]["weight"] += 1
        if rgb_features[granule_index] == 2 and d_features[granule_index] == 2:
            graph["BeR"]["BeD"]["weight"] += 1

        if d_features[granule_index] == 0 and background_granules[granule_index] == 1:
            graph["NBD"]["D2"]["weight"] += 1
        if d_features[granule_index] == 0 and object_granules[granule_index] == 1:
            graph["NBD"]["D1"]["weight"] += 1
        if d_features[granule_index] == 3 and object_granules[granule_index] == 1:
            graph["CCD"]["D1"]["weight"] += 1
        if d_features[granule_index] == 1 and object_granules[granule_index] == 1:
            graph["PBD"]["D1"]["weight"] += 1
        if d_features[granule_index] == 2 and object_granules[granule_index] == 1:
            graph["BeD"]["D1"]["weight"] += 1

