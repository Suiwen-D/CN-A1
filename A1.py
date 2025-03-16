import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

if len(sys.argv) < 2:
    print("Pass the path to the network as an argument")
    exit(-2)

path = sys.argv[1]
if not path:
    print("Pass the path to the network as an argument")
    exit(-1)

def analyze_network(net_path):
    # Read the network
    G = nx.read_pajek(net_path)

    # Check if it is a multi-graph, if so, convert it to a simple graph
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        if isinstance(G, nx.MultiGraph):
            # Convert multiple undirected graphs to simple undirected graphs
            new_G = nx.Graph()
            for u, v in G.edges():
                if not new_G.has_edge(u, v):
                    new_G.add_edge(u, v)
            G = new_G
        else:
            # Convert Multiple Directional Graphs to Simple Directional Graphs
            new_G = nx.DiGraph()
            for u, v in G.edges():
                if not new_G.has_edge(u, v):
                    new_G.add_edge(u, v)
            G = new_G

    # Basic Network Indicators
    basic_stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'degrees': [d for n, d in G.degree()],
        'min_degree': min(d for n, d in G.degree()),
        'max_degree': max(d for n, d in G.degree()),
        'avg_degree': np.mean([d for n, d in G.degree()])
    }

    # Average clustering coefficient
    basic_stats['avg_clustering'] = nx.average_clustering(G)

    # Degree Correlation
    basic_stats['assortativity'] = nx.degree_assortativity_coefficient(G)

    # Connectivity Processing
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc)
        basic_stats['avg_path_length'] = nx.average_shortest_path_length(G_lcc)
        basic_stats['diameter'] = nx.diameter(G_lcc)
    else:
        basic_stats['avg_path_length'] = nx.average_shortest_path_length(G)
        basic_stats['diameter'] = nx.diameter(G)

    # Degree Distribution Analysis 
    degrees = np.array(basic_stats['degrees'])
    k_min = np.min(degrees)
    k_max = np.max(degrees)

    if k_min == 0:
        k_min = 1  # avoid log(0)

    log_k_min = np.log10(k_min)
    log_k_max = np.log10(k_max + 1)
    num_bins = 10
    log_bins = np.linspace(log_k_min, log_k_max, num_bins + 1)
    bins = 10 ** log_bins

    counts, _ = np.histogram(degrees, bins=bins)
    probabilities = counts / len(degrees)
    ccdf = np.cumsum(probabilities[::-1])[::-1]

    bin_centers = 10 ** ((log_bins[:-1] + log_bins[1:]) / 2)

    # Central Analysis
    centralities = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G)
    }

    # Dealing with possible convergence of eigenvector centrality
    try:
        centralities['eigenvector'] = nx.eigenvector_centrality(G)
    except nx.PowerIterationFailedConvergence:
        centralities['eigenvector'] = nx.eigenvector_centrality_numpy(G)

    # Extract the top five
    top_central = {}
    for metric in centralities:
        sorted_nodes = sorted(centralities[metric].items(), key=lambda x: x[1], reverse=True)[:5]
        top_central[metric] = [(n, round(v, 4)) for n, v in sorted_nodes]

    return {
        'basic_stats': basic_stats,
        'degree_distribution': (bin_centers, ccdf),
        'top_central': top_central
    }

# I tried to deal with all the net documents, but always failed, so only 1 net document per time, we can change the net document name to analyze different net documents
results = analyze_network(path)

# Printing Basic Statistics
print("Network Statistics:")
print(f"Nodes: {results['basic_stats']['nodes']}")
print(f"Edges: {results['basic_stats']['edges']}")
print(f"Min Degree: {results['basic_stats']['min_degree']}")
print(f"Max Degree: {results['basic_stats']['max_degree']}")
print(f"Avg Degree: {results['basic_stats']['avg_degree']:.2f}")
print(f"Avg Clustering: {results['basic_stats']['avg_clustering']:.4f}")
print(f"Assortativity: {results['basic_stats']['assortativity']:.4f}")
print(f"Avg Path Length: {results['basic_stats']['avg_path_length']:.2f}")
print(f"Diameter: {results['basic_stats']['diameter']}")

# Distribution of degree
plt.figure(figsize=(10, 6))
plt.loglog(results['degree_distribution'][0], results['degree_distribution'][1], 'bo-')
plt.title('Degree Distribution (CCDF in log-log scale)')
plt.xlabel('log(k)')
plt.ylabel('log(CCDF(k))')
plt.grid(True)
plt.show()

# Display Central Results
print("\nTop Central Nodes:")
for metric, nodes in results['top_central'].items():
    print(f"\n{metric.capitalize()}:")
    for node, score in nodes:
        print(f"Node {node}: {score}")
