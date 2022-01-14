import networkx as nx


def load_graph(graph_name: str):
    if graph_name == "karate":
        return nx.karate_club_graph()
