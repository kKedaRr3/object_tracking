import numpy as np

class FlowGraph:
    """
    Zaimplementowany jako słownik:
    nodes: { node_id: value }
    edges: { (node_id1, node_id2): flow_val }
    """
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, val=0):
        self.nodes[node_id] = self.nodes.get(node_id, 0) + val

    def add_edge(self, from_node, to_node, val=0):
        self.edges[(from_node, to_node)] = self.edges.get((from_node, to_node), 0) + val

    def total_flow(self):
        return sum(self.nodes.values())

    def normalize(self):
        total = self.total_flow()
        if total > 0:
            for k in self.nodes:
                self.nodes[k] /= total
            for k in self.edges:
                self.edges[k] /= total

    def expected_flow_graph(self, new_class_dist):
        """
        Tworzy przewidywany graf przepływu na podstawie nowego rozkładu klas new_class_dist
        new_class_dist np. dictionary -> {"classA": alpha, "classB": beta, ...}
        Zależnie od definicji w pracy.
        """
        # Bardzo uproszczony przykład
        new_fg = FlowGraph()
        for node_id, val in self.nodes.items():
            # ewentualne przeskalowanie w zależności od new_class_dist
            new_val = val # * jakiś współczynnik
            new_fg.add_node(node_id, new_val)
        for (f,t), w in self.edges.items():
            # Również przeskalować
            new_w = w # * ...
            new_fg.add_edge(f,t,new_w)
        return new_fg

def mutual_dependency(flowg1, flowg2, eps=1e-7):
    """
    Oblicza 'µ = NF G1 / NF G2' z pracy – tu symbolicznie.
    Zakładamy, że flowg1 i flowg2 mają te same węzły i krawędzie
    """
    mu_values = []
    for n in flowg1.nodes:
        val1 = flowg1.nodes[n]
        val2 = flowg2.nodes.get(n, 0)
        if val2 < eps:
            ratio = 999999
        else:
            ratio = val1 / val2
        mu_values.append(ratio)
    # np. zwracamy średnią
    return np.mean(mu_values)
