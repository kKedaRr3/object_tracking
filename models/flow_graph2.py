import networkx as nx
from utils.visualization import Visualization

def generate_empty_graph():
    graph = nx.DiGraph()

    graph.add_nodes_from(["O", "B"])
    graph.add_nodes_from(["NBT", "CCT", "PBT", "BeT"])
    graph.add_nodes_from(["NBR", "PBR", "BeR"])
    graph.add_nodes_from(["NBD", "CCD", "PBD", "BeD"])
    graph.add_nodes_from(["D1", "D2"])

    graph.add_edges_from([("O", "NBT"), ("O", "CCT"), ("O", "PBT"), ("O", "BeT")])
    graph.add_edges_from([("B", "NBT"), ("B", "PBT"), ("O", "BeT")])

    graph.add_edges_from([("NBT", "NBR"), ("NBT", "BeR")])
    graph.add_edges_from([("CCT", "BeR")])
    graph.add_edges_from([("PBT", "PBR"), ("PBT", "BeR")])
    graph.add_edges_from([("BeT", "NBR"), ("BeT", "BeR")])

    graph.add_edges_from([("NBR", "NBD"), ("NBR", "BeD")])
    graph.add_edges_from([("PBR", "CCD"), ("PBR", "PBD")])
    graph.add_edges_from([("BeR", "PBD"), ("BeR", "BeD")])

    graph.add_edges_from([("NBD", "D1"), ("NBD", "D2")])
    graph.add_edges_from([("CCD", "D1")])
    graph.add_edges_from([("PBD", "D1")])
    graph.add_edges_from([("BeD", "D1")])

    return graph

# Aby obliczyc wagi na krawedzich trzeba sprawdzic ile jest jednoczesnie takich granul do kotrych jest podlaczona krawedz
# To znaczy majac krawedz k(BeT, BeR) sprawdzamy ile jest granul ktore sa BeT i BeR jednoczesnie
def add_weights_to_flow_graph(graph, rule_base, features):
    # TODO policzyc ile jest wszstkich granul kazdego rodzaju i z kazdym atrybutem (Be, NB, PB, CC)
    sp_t_features, rgb_features, d_features = features
    height, width = rule_base.shape
    for y in range(height):
        for x in range(width):
            pass


