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

graph = generate_empty_graph()
Visualization.draw_flow_graph(graph)
