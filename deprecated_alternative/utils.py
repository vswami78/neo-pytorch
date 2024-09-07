import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from typing import Any, List, Tuple, Callable
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DebugInfo:
    original_node_counts: Dict[str, int] = field(default_factory=dict)
    filtered_node_counts: Dict[str, int] = field(default_factory=dict)
    filter_conditions: Dict[str, List[str]] = field(default_factory=dict)
    combine_method: str = ""
    messages: List[str] = field(default_factory=list)

    def add_message(self, message: str):
        self.messages.append(message)

    def __str__(self):
        return (
            f"Original node counts: {self.original_node_counts}\n"
            f"Filtered node counts: {self.filtered_node_counts}\n"
            f"Filter conditions: {self.filter_conditions}\n"
            f"Combine method: {self.combine_method}\n"
            f"Debug messages:\n" + "\n".join(self.messages)
        )

def attribute_based_filter(graph: HeteroData, node_type: str, attribute: str, value: Any) -> torch.Tensor:
    if attribute not in graph[node_type].keys():
        raise ValueError(f"Attribute '{attribute}' not found for node type '{node_type}'")
    return graph[node_type][attribute] == value

def find_most_popular_product(graph: HeteroData) -> str:
    edge_index = graph["user", "purchased", "product"].edge_index
    product_counts = torch.bincount(edge_index[1])
    most_popular_index = torch.argmax(product_counts).item()
    return graph["product"].product_id[most_popular_index]

def find_users_who_purchased_product(graph: HeteroData, product_id: str) -> List[str]:
    edge_index = graph["user", "purchased", "product"].edge_index
    product_mask = graph["product"].product_id == product_id
    product_node_index = torch.where(torch.tensor(product_mask))[0]
    user_indices = edge_index[0][edge_index[1] == product_node_index]
    return graph["user"].user_id[user_indices].tolist()

def visualize_subgraph(subgraph, node_color_map, edge_color_map):
    """
    Visualize the subgraph, highlighting only relevant user-product-category connections.
    """
    debug_info = DebugInfo()
    debug_info.add_message(f"Subgraph node counts: {[subgraph[nt].num_nodes for nt in subgraph.node_types]}")
    if subgraph['user'].num_nodes == 0:
        debug_info.add_message("No users meet the filter condition. Nothing to display.")
        return None, debug_info

    fig, ax = plt.subplots(figsize=(12, 8))
    G = nx.Graph()
    
    # Add user nodes
    for i in range(subgraph['user'].num_nodes):
        G.add_node(f"user_{subgraph['user'].user_id[i]}", type='user')

    # Add only purchased products and their categories
    user_product_edges = subgraph['user', 'purchased', 'product'].edge_index.t().tolist()
    relevant_products = set()
    for user_idx, product_idx in user_product_edges:
        user_id = subgraph['user'].user_id[user_idx]
        product_id = subgraph['product'].product_id[product_idx]
        G.add_node(f"product_{product_id}", type='product')
        G.add_edge(f"user_{user_id}", f"product_{product_id}")
        relevant_products.add(product_idx)

    # Add categories of relevant products
    product_category_edges = subgraph['product', 'belongs_to', 'category'].edge_index.t().tolist()
    for product_idx, category_idx in product_category_edges:
        if product_idx in relevant_products:
            product_id = subgraph['product'].product_id[product_idx]
            category_id = subgraph['category'].category_id[category_idx]
            G.add_node(f"category_{category_id}", type='category')
            G.add_edge(f"product_{product_id}", f"category_{category_id}")

    # Assign colors
    node_colors = [node_color_map[G.nodes[node]['type']] for node in G.nodes()]
    edge_colors = [edge_color_map[(G.nodes[u]['type'], G.nodes[v]['type'])] for u, v in G.edges()]

    # Draw the graph
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust layout parameters
    nx.draw(G, pos, ax=ax, node_color=node_colors, edge_color=edge_colors, with_labels=False, node_size=300)
    
    # Add labels with original IDs
    labels = {node: node.split('_')[1] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{k.capitalize()}s',
                                  markerfacecolor=v, markersize=10) for k, v in node_color_map.items()]
    legend_elements += [plt.Line2D([0], [0], color=v, label=f'{k[0].capitalize()}-{k[1].capitalize()}')
                        for k, v in edge_color_map.items()]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_title("Subgraph Visualization (Relevant Connections Only)")
    
    # Adjust the layout without using tight_layout
    plt.subplots_adjust(right=0.8)  # Adjust this value as needed
    
    return fig, debug_info

def extract_subgraph(graph, filters):
    """
    Extract a subgraph based on user-defined filters, including only relevant products and categories.
    
    :param graph: PyTorch Geometric HeteroData object
    :param filters: Dictionary of node types and their filter conditions
    :return: Extracted subgraph, debug_info
    """
    node_mask = {node_type: torch.zeros(graph[node_type].num_nodes, dtype=torch.bool) for node_type in graph.node_types}
    debug_info = DebugInfo()
    
    # Apply user filter
    user_mask = filters['user'](graph['user'])
    node_mask['user'] = user_mask
    debug_info.add_message(f"User mask sum: {user_mask.sum().item()}")
    
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
    
    debug_info.filtered_node_counts = {nt: mask.sum().item() for nt, mask in node_mask.items()}
    debug_info.filter_conditions = {nt: str(filters[nt]) for nt in filters}
    debug_info.combine_method = "AND"  # Since this function doesn't support OR
    
    debug_info.add_message(f"Final node mask sums: {[mask.sum().item() for mask in node_mask.values()]}")
    
    return graph.subgraph(node_mask), debug_info

def extract_subgraph_with_multiple_conditions(graph: HeteroData, filters: Dict[str, List[Callable]],
                                              combine_with_or: bool = False) -> Tuple[HeteroData, DebugInfo]:
    debug_info = DebugInfo()
    node_mask = {node_type: torch.ones(graph[node_type].num_nodes, dtype=torch.bool)
                 for node_type in graph.node_types}

    for node_type, conditions in filters.items():
        debug_info.add_message(f"Processing node type: {node_type}")
        debug_info.add_message(f"Available keys: {graph[node_type].keys()}")
        debug_info.add_message(f"Number of nodes: {graph[node_type].num_nodes}")
        
        if 'x' in graph[node_type]:
            debug_info.add_message(f"Shape of x: {graph[node_type].x.shape}")
        
        if f'{node_type}_id' in graph[node_type]:
            debug_info.add_message(f"First few {node_type} IDs: {graph[node_type][f'{node_type}_id'][:5]}")
        
        node_type_mask = torch.ones(graph[node_type].num_nodes, dtype=torch.bool)
        for condition in conditions:
            try:
                condition_mask = condition(graph[node_type])
                if combine_with_or:
                    node_type_mask |= condition_mask
                else:
                    node_type_mask &= condition_mask
            except Exception as e:
                debug_info.add_message(f"Error applying condition: {str(e)}")
                debug_info.add_message(f"Condition: {condition}")
        node_mask[node_type] = node_type_mask

    node_mask = {node_type: mask.bool() for node_type, mask in node_mask.items()}
    subgraph = graph.subgraph(node_mask)

    debug_info.original_node_counts = {nt: graph[nt].num_nodes for nt in graph.node_types}
    debug_info.filtered_node_counts = {nt: subgraph[nt].num_nodes for nt in subgraph.node_types}
    debug_info.filter_conditions = {nt: [str(cond) for cond in conditions] for nt, conditions in filters.items()}
    debug_info.combine_method = "OR" if combine_with_or else "AND"

    return subgraph, debug_info