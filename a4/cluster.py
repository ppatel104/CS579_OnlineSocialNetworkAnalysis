"""
cluster.py
"""
from collections import Counter, defaultdict, deque
import math
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def community_detection(G,k):
    """
    :param G: graph
    :param k: number of clusters
    :return: resulting clusters as tuples
    """
    comp = girvan_newman(G)

    result = tuple
    for comp in itertools.islice(comp, k-1):

        result = tuple(sorted(c) for c in comp)

    return result

def girvan_newman(G, most_valuable_edge=None):
    '''
    This code is an implementation of girvan_newman algorithm provided by networkx in their documentation
    https://networkx.github.io/documentation/latest/_modules/networkx/algorithms/community/centrality.html#girvan_newman
    :param G:
    :param most_valuable_edge:
    :return:
    '''
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return

    if most_valuable_edge is None:
        def most_valuable_edge(G):
            betweenness = nx.edge_betweenness_centrality(G)
            return max(betweenness, key=betweenness.get)
    g = G.copy().to_undirected()
    g.remove_edges_from(g.selfloop_edges())
    # print("\t\t********************* - Completed Clustering of Graph - *********************")
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)


def _without_most_central_edges(G, most_valuable_edge):
    '''
    params: graph, and the most_valuable_edge
    returns: new componenets of the graph.
    '''
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components

def read_graph():
    """ Read 'friends.txt' into a networkx **undirected** graph.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('friends.txt', delimiter='\t')

def draw_graph(graph):
    figure = plt.figure()
    Graph = nx.draw_networkx(graph, with_labels=False,pos=nx.spring_layout(graph), node_size=40, alpha=0.40, width=0.12, font_size=8,labels=None)
    plt.axis("off")
    plt.savefig("network.png")

def draw_clusters(graph,t):
    clusters = {}
    for i in range(0,len(t)):
        clusters[i] = t[i]
    g = graph
    n_c = ['g' if n in clusters[1] else 'b' for n in g.nodes()]
    figure = plt.figure()
    Graph = nx.draw_networkx(graph, with_labels=False, pos=nx.spring_layout(graph), node_size=40, alpha=0.40,width=0.12, font_size=8, node_color=n_c)
    plt.axis("off")
    plt.savefig("clusters.png")

def main():
    #download_data()
    graph = read_graph()
    draw_graph(graph)
    t = community_detection(G=graph, k=2)
    print ("Number of communities discovered: %d"%len(t))
    print ("Average number of users per community: %d"%((len(t[0])+len(t[1]))/len(t)))
    draw_clusters(graph,t)
    f = open('cluster.txt',"w")
    f.write('%d\n'%len(t))
    f.write('%d'%((len(t[0])+len(t[1]))/len(t)))
    f.close()

if __name__ == '__main__':
    print ("---------------cluster.py----------------------")
    main()
    print ("----------------cluster.py-------------------------")
