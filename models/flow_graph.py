import networkx as nx
import numpy as np


# TODO
# Do przeanalizowania/zrobienia:
# 1) W rule base jest duzo niezakwalifikowanych granul ktore nie sa ani obiektem ani tlem aczkolwiek wydaje sie ze powinny byc tlem


def generate_flow_graph(features):
    flow_graph = generate_empty_graph()
    add_weights_to_flow_graph(flow_graph, features)
    return flow_graph


def compute_rule_base_coverage(g1, features):
    nfg1 = compute_normalized_graph(g1)

    g2 = generate_flow_graph(features)
    nfg2 = compute_normalized_graph(g2)

    nfg1 = compute_expected_graph(nfg1, nfg2)

    return compute_mutual_dependency(nfg1, nfg2), nfg2


def compute_normalized_graph(fg2):
    all_granules_count = fg2.nodes["O"]["weight"] + fg2.nodes["B"]["weight"]
    for _, _, data in fg2.edges(data=True):
        data["weight"] = data["weight"] / all_granules_count
    for _, data in fg2.nodes(data=True):
        data["weight"] = data["weight"] / all_granules_count
    return fg2


def compute_expected_graph(nfg1, nfg2):
    for u, v, data in nfg1.edges(data=True):
        if nfg1.nodes[u]["weight"] == 0: continue
        alfa = nfg2.nodes[u]["weight"]
        data["weight"] = alfa * data["weight"] / nfg1.nodes[u]["weight"]
    for node in nfg1.nodes():
        nfg1.nodes[node]["weight"] = nfg2.nodes[node]["weight"]
    return nfg1


def compute_mutual_dependency(fg1, fg2):
    sum_top, sum_bottom = 0, 0
    for u, v, data in fg1.edges(data=True):
        sum_top += np.abs(data["weight"] - fg2[u][v]["weight"])
        sum_bottom += data["weight"]
    return sum_top / sum_bottom


def get_features_to_update(g1, g2, thr):
    features_to_update = []
    for u, v, data in g1.edges(data=True):
        if g2[u][v]["weight"] == 0 and data["weight"] == 0:
            dependency = 0
        elif g2[u][v]["weight"] == 0:
            dependency = np.abs(data["weight"] - 1)
        else:
            dependency = np.abs(data["weight"] / g2[u][v]["weight"] - 1)
        if dependency > thr:
            if v.endswith("T"):
                features_to_update.append("sp_t")
            if v.endswith("R"):
                features_to_update.append("rgb")
            if v.endswith("D"):
                features_to_update.append("d")

    return features_to_update


def generate_empty_graph():
    graph = nx.DiGraph()

    graph.add_nodes_from(["O", "B"], weight=0)
    graph.add_nodes_from(["NBT", "CCT", "PBT", "BeT"], weight=0)
    graph.add_nodes_from(["NBR", "PBR", "BeR"], weight=0)
    graph.add_nodes_from(["NBD", "CCD", "PBD", "BeD"], weight=0)
    graph.add_nodes_from(["D1", "D2"], weight=0)

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

        if object_granules[granule_index] == 1:
            graph.nodes["O"]["weight"] += 1
        if background_granules[granule_index] == 1:
            graph.nodes["B"]["weight"] += 1
        if sp_t_features[granule_index] == 0:
            graph.nodes["NBT"]["weight"] += 1
        if sp_t_features[granule_index] == 1:
            graph.nodes["PBT"]["weight"] += 1
        if sp_t_features[granule_index] == 2:
            graph.nodes["BeT"]["weight"] += 1
        if sp_t_features[granule_index] == 3:
            graph.nodes["CCT"]["weight"] += 1
        if rgb_features[granule_index] == 0:
            graph.nodes["NBR"]["weight"] += 1
        if rgb_features[granule_index] == 1:
            graph.nodes["PBR"]["weight"] += 1
        if rgb_features[granule_index] == 2:
            graph.nodes["BeR"]["weight"] += 1
        if d_features[granule_index] == 0:
            graph.nodes["NBD"]["weight"] += 1
        if d_features[granule_index] == 1:
            graph.nodes["PBD"]["weight"] += 1
        if d_features[granule_index] == 2:
            graph.nodes["BeD"]["weight"] += 1
        if d_features[granule_index] == 3:
            graph.nodes["CCD"]["weight"] += 1

        # - detection
        if background_granules[granule_index] == 1 and d_features[granule_index] == 0 and rgb_features[
            granule_index] == 0 and sp_t_features[granule_index] == 2:
            graph.nodes["D2"]["weight"] += 1
        # + detection
        else:
            graph.nodes["D1"]["weight"] += 1

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

        if background_granules[granule_index] == 1 and sp_t_features[granule_index] == 0 and rgb_features[
            granule_index] == 0 and d_features[granule_index] == 0:
            graph["NBD"]["D1"]["weight"] += 1
        if background_granules[granule_index] == 1 and sp_t_features[granule_index] == 0 and rgb_features[
            granule_index] == 2 and d_features[granule_index] == 2:
            graph["BeD"]["D1"]["weight"] += 1
        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 1 and rgb_features[
            granule_index] == 2 and d_features[granule_index] == 2:
            graph["BeD"]["D1"]["weight"] += 1
        if background_granules[granule_index] == 1 and sp_t_features[granule_index] == 1 and rgb_features[
            granule_index] == 2 and d_features[granule_index] == 1:
            graph["PBD"]["D1"]["weight"] += 1
        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 2 and rgb_features[
            granule_index] == 2 and d_features[granule_index] == 1:
            graph["PBD"]["D1"]["weight"] += 1
        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 3 and rgb_features[
            granule_index] == 2 and d_features[granule_index] == 2:
            graph["BeD"]["D1"]["weight"] += 1
        if background_granules[granule_index] == 1 and sp_t_features[granule_index] == 0 and rgb_features[
            granule_index] == 0 and d_features[granule_index] == 2:
            graph["BeD"]["D1"]["weight"] += 1
        if background_granules[granule_index] == 1 and d_features[granule_index] == 0 and rgb_features[
            granule_index] == 0 and sp_t_features[granule_index] == 2:
            graph["NBD"]["D2"]["weight"] += 1
        if object_granules[granule_index] == 1 and d_features[granule_index] == 0 and rgb_features[
            granule_index] == 0 and sp_t_features[granule_index] == 2:
            graph["NBD"]["D1"]["weight"] += 1
        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 1 and rgb_features[
            granule_index] == 1 and d_features[granule_index] == 3:
            graph["CCD"]["D1"]["weight"] += 1
        if object_granules[granule_index] == 1 and sp_t_features[granule_index] == 2 and rgb_features[
            granule_index] == 2 and d_features[granule_index] == 2:
            graph["BeD"]["D1"]["weight"] += 1
