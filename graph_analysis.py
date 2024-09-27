import torch
from torch_geometric.data import HeteroData
from collections import Counter

def analyze_graph(hetero_data: HeteroData):
    """
    Perform various analyses on the PyTorch Geometric heterogeneous graph.
    
    :param hetero_data: PyTorch Geometric HeteroData object
    :return: Dictionary containing analysis results
    """
    results = {}
    
    # Basic graph properties
    results['num_nodes'] = sum(hetero_data[node_type].num_nodes for node_type in hetero_data.node_types)
    results['num_edges'] = sum(hetero_data[edge_type].num_edges for edge_type in hetero_data.edge_types)
    
    # Node type distribution
    results['node_type_distribution'] = {node_type: hetero_data[node_type].num_nodes for node_type in hetero_data.node_types}
    
    # Edge type distribution
    results['edge_type_distribution'] = {edge_type: hetero_data[edge_type].num_edges for edge_type in hetero_data.edge_types}
    
    # Degree distribution
    degree_distribution = {}
    for edge_type in hetero_data.edge_types:
        src, _, dst = edge_type
        src_degrees = torch.bincount(hetero_data[edge_type].edge_index[0])
        dst_degrees = torch.bincount(hetero_data[edge_type].edge_index[1])
        degree_distribution[f"{src}_out"] = src_degrees.tolist()
        degree_distribution[f"{dst}_in"] = dst_degrees.tolist()
    results['degree_distribution'] = degree_distribution
    
    # Average degree
    results['avg_degree'] = {
        node_type: sum(degree_distribution[f"{node_type}_out"]) / hetero_data[node_type].num_nodes
        for node_type in hetero_data.node_types if f"{node_type}_out" in degree_distribution
    }
    
    # Connected components analysis is not straightforward in PyG, 
    # we'll skip it for now as it would require converting to NetworkX
    
    return results

def generate_analysis_report(analysis_results):
    """
    Generate a formatted report from the analysis results.
    
    :param analysis_results: Dictionary containing analysis results
    :return: Formatted string report
    """
    report = "Graph Analysis Report\n"
    report += "=====================\n\n"
    
    report += f"Total number of nodes: {analysis_results['num_nodes']}\n"
    report += f"Total number of edges: {analysis_results['num_edges']}\n\n"
    
    report += "Node Type Distribution:\n"
    for node_type, count in analysis_results['node_type_distribution'].items():
        report += f"  {node_type}: {count}\n"
    report += "\n"
    
    report += "Edge Type Distribution:\n"
    for edge_type, count in analysis_results['edge_type_distribution'].items():
        report += f"  {edge_type}: {count}\n"
    report += "\n"
    
    report += "Average Degree:\n"
    for node_type, avg_degree in analysis_results['avg_degree'].items():
        report += f"  {node_type}: {avg_degree:.2f}\n"
    report += "\n"
    
    report += "Degree Distribution Summary:\n"
    for key, degrees in analysis_results['degree_distribution'].items():
        non_zero_degrees = [d for d in degrees if d > 0]
        if non_zero_degrees:
            report += f"  {key}:\n"
            report += f"    Min: {min(non_zero_degrees)}, Max: {max(non_zero_degrees)}, "
            report += f"Median: {sorted(non_zero_degrees)[len(non_zero_degrees)//2]}\n"
    
    return report