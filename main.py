import pandas as pd
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt

# Constants
NODE_COLOR_MAP = {"user": "red", "product": "blue", "category": "green"}
EDGE_COLOR_MAP = {("user", "product"): "orange", ("product", "category"): "purple"}

def create_heterogeneous_graph(data_path, node_types, edge_types):
    """
    Create a heterogeneous graph from tabular data.
    
    :param data_path: Path to the CSV file containing the tabular data
    :param node_types: Dictionary mapping node types to their identifying columns
    :param edge_types: List of tuples defining edge types (source, edge_type, target)
    :return: PyTorch Geometric HeteroData object
    """
    df = pd.read_csv(data_path)
    data = HeteroData()
    
    # Create nodes
    for node_type, id_col in node_types.items():
        unique_ids = df[id_col].unique()
        data[node_type].x = torch.arange(len(unique_ids))
        data[node_type].mapping = {id: i for i, id in enumerate(unique_ids)}
        data[node_type].original_id = unique_ids  # Store original IDs as a numpy array
    
    # Create edges
    for src_type, edge_type, dst_type in edge_types:
        src_ids = df[node_types[src_type]].map(data[src_type].mapping)
        dst_ids = df[node_types[dst_type]].map(data[dst_type].mapping)
        edge_index = torch.stack([torch.tensor(src_ids.values), torch.tensor(dst_ids.values)], dim=0)
        data[src_type, edge_type, dst_type].edge_index = edge_index
    
    return data

def extract_subgraph(graph, filters):
    """
    Extract a subgraph based on user-defined filters, including only relevant products and categories.
    
    :param graph: PyTorch Geometric HeteroData object
    :param filters: Dictionary of node types and their filter conditions
    :return: Extracted subgraph, debug_info
    """
    node_mask = {node_type: torch.zeros(graph[node_type].num_nodes, dtype=torch.bool) for node_type in graph.node_types}
    debug_info = {}
    
    # Apply user filter
    user_mask = filters['user'](graph['user'])
    node_mask['user'] = user_mask
    debug_info['user'] = {
        'total_nodes': graph['user'].num_nodes,
        'nodes_passing_filter': user_mask.sum().item(),
        'filter_condition': str(filters['user'])
    }
    
    # Get products purchased by filtered users
    user_product_edges = graph['user', 'purchased', 'product'].edge_index
    filtered_user_indices = torch.where(user_mask)[0]
    relevant_product_mask = torch.isin(user_product_edges[0], filtered_user_indices)
    relevant_product_indices = user_product_edges[1][relevant_product_mask]
    node_mask['product'][relevant_product_indices] = True
    
    # Get categories of the relevant products
    product_category_edges = graph['product', 'belongs_to', 'category'].edge_index
    relevant_category_mask = torch.isin(product_category_edges[0], relevant_product_indices)
    relevant_category_indices = product_category_edges[1][relevant_category_mask]
    node_mask['category'][relevant_category_indices] = True
    
    debug_info['product'] = {
        'total_nodes': graph['product'].num_nodes,
        'nodes_passing_filter': node_mask['product'].sum().item(),
        'filter_condition': 'Products purchased by filtered users'
    }
    debug_info['category'] = {
        'total_nodes': graph['category'].num_nodes,
        'nodes_passing_filter': node_mask['category'].sum().item(),
        'filter_condition': 'Categories of products purchased by filtered users'
    }
    
    return graph.subgraph(node_mask), debug_info

def visualize_subgraph(subgraph, debug_info):
    """
    Visualize the subgraph, highlighting user-product connections.
    If no users meet the filter condition, don't display anything.
    
    :param subgraph: PyTorch Geometric HeteroData object (subgraph)
    :param debug_info: Dictionary containing debugging information
    """
    # Print debugging information
    print("Debugging Information:")
    for node_type, info in debug_info.items():
        print(f"{node_type.capitalize()}:")
        print(f"  Total nodes: {info['total_nodes']}")
        print(f"  Nodes passing filter: {info['nodes_passing_filter']}")
        print(f"  Filter condition: {info['filter_condition']}")
        print()

    # Check if there are any users in the subgraph
    if subgraph['user'].num_nodes == 0:
        print("No users meet the filter condition. Nothing to display.")
        return

    G = nx.Graph()
    
    # Add nodes with original IDs
    for node_type in subgraph.node_types:
        original_ids = subgraph[node_type].original_id
        nodes = [(f"{node_type}_{original_ids[i]}", {"type": node_type}) for i in range(subgraph[node_type].num_nodes)]
        G.add_nodes_from(nodes)
    
    # Add edges
    for edge_type in subgraph.edge_types:
        edge_index = subgraph[edge_type].edge_index.t().tolist()
        src_type, _, dst_type = edge_type
        edges = [(f"{src_type}_{subgraph[src_type].original_id[src]}", f"{dst_type}_{subgraph[dst_type].original_id[dst]}") for src, dst in edge_index]
        G.add_edges_from(edges)
    
    # Debug print statements (commented out)
    # print(f"Number of nodes: {len(G.nodes())}")
    # print(f"Number of edges: {len(G.edges())}")
    # print("Node types:")
    # for node in G.nodes():
    #     print(f"  {node}: {G.nodes[node]['type']}")

    # Assign colors
    node_colors = [NODE_COLOR_MAP[G.nodes[node]['type']] for node in G.nodes()]
    edge_colors = [EDGE_COLOR_MAP[(G.nodes[u]['type'], G.nodes[v]['type'])] for u, v in G.edges()]

    # Add these debug print statements (commented out)
    # print(f"Number of nodes: {len(G.nodes())}")
    # print(f"Number of colors: {len(node_colors)}")
    # print(f"Number of edges: {len(G.edges())}")
    # print(f"Number of edge colors: {len(edge_colors)}")

    pos = nx.spring_layout(G)  
    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True, node_size=500, font_size=8)
    
    # Add labels with original IDs
    labels = {node: node.split('_')[1] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{k.capitalize()}s',
                                  markerfacecolor=v, markersize=10) for k, v in NODE_COLOR_MAP.items()]
    legend_elements += [plt.Line2D([0], [0], color=v, label=f'{k[0].capitalize()}-{k[1].capitalize()}')
                        for k, v in EDGE_COLOR_MAP.items()]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("Subgraph Visualization")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example data and parameters
    data_path = "user-prod-categ.csv"
    node_types = {
        "user": "user_id",
        "product": "product_id",
        "category": "category_id"
    }
    edge_types = [
        ("user", "purchased", "product"),
        ("product", "belongs_to", "category")
    ]
    
    # Create heterogeneous graph
    graph = create_heterogeneous_graph(data_path, node_types, edge_types)
    
    # Define filters (example: users who made more than 2 purchases)
    filters = {
        "user": lambda x: torch.bincount(graph["user", "purchased", "product"].edge_index[0]) > 2
    }
    
    # Extract subgraph
    subgraph, debug_info = extract_subgraph(graph, filters)
    
    # Visualize subgraph
    visualize_subgraph(subgraph, debug_info)
