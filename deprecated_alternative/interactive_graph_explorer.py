import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import torch
from torch_geometric.data import HeteroData
import logging
from debug_info import DebugInfo

from types import SimpleNamespace
import matplotlib.pyplot as plt

# from graph_visualization import GraphVisualizer

# # Set up logging
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

class InteractiveGraphExplorer:
    def __init__(self, hetero_data: HeteroData, logger: logging.Logger, config: SimpleNamespace):
        self.hetero_data = hetero_data
        self.logger = logger
        self.config = config
        
        self.node_color_map = config.node_color_map
        self.edge_color_map = config.edge_color_map
        
        # Create reverse mappings for each node type
        self.reverse_mappings = {}
        for node_type, id_field in self.config.node_types.items():
            hashed_ids = self.hetero_data[node_type][id_field]
            original_ids = self.hetero_data[node_type].original_ids
            self.reverse_mappings[node_type] = dict(zip(hashed_ids.tolist(), original_ids))
        
        self.G = None
        self.pos = None
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

    def get_node_id(self, node_type, index):
        id_field = self.config.node_types[node_type]
        hashed_id = self.hetero_data[node_type][id_field][index].item()
        return self.reverse_mappings[node_type][hashed_id]

    def create_networkx_graph(self, seed_node_type, seed_node_id):
        self.G = nx.Graph()
        
        id_field = self.config.node_types[seed_node_type]
        hashed_seed_id = hash(seed_node_id)
        seed_index = (self.hetero_data[seed_node_type][id_field] == hashed_seed_id).nonzero(as_tuple=True)[0]
        
        if len(seed_index) == 0:
            self.logger.error(f"Seed node {seed_node_id} not found in {seed_node_type} nodes.")
            return

        seed_index = seed_index[0].item()
        self.G.add_node(seed_node_id, node_type=seed_node_type)
        self.expand_graph(seed_node_type, seed_node_id)

    def expand_graph(self, node_type, node_id):
        id_field = self.config.node_types[node_type]
        hashed_node_id = hash(node_id)

        for edge_type in self.hetero_data.edge_types:
            src, rel, dst = edge_type
            edge_index = self.hetero_data[edge_type].edge_index

            if src == node_type:
                src_mask = self.hetero_data[src][id_field] == hashed_node_id
                src_indices = src_mask.nonzero(as_tuple=True)[0]
                for src_idx in src_indices:
                    dst_indices = edge_index[1][edge_index[0] == src_idx]
                    for dst_idx in dst_indices:
                        hashed_dst_id = self.hetero_data[dst][self.config.node_types[dst]][dst_idx].item()
                        dst_id = self.reverse_mappings[dst][hashed_dst_id]
                        if dst_id not in self.G:
                            self.G.add_node(dst_id, node_type=dst)
                        if not self.G.has_edge(node_id, dst_id):
                            self.G.add_edge(node_id, dst_id, edge_type=f"{src}_{dst}")

            elif dst == node_type:
                dst_mask = self.hetero_data[dst][id_field] == hashed_node_id
                dst_indices = dst_mask.nonzero(as_tuple=True)[0]
                for dst_idx in dst_indices:
                    src_indices = edge_index[0][edge_index[1] == dst_idx]
                    for src_idx in src_indices:
                        hashed_src_id = self.hetero_data[src][self.config.node_types[src]][src_idx].item()
                        src_id = self.reverse_mappings[src][hashed_src_id]
                        if src_id not in self.G:
                            self.G.add_node(src_id, node_type=src)
                        if not self.G.has_edge(src_id, node_id):
                            self.G.add_edge(src_id, node_id, edge_type=f"{src}_{dst}")

    def draw_graph(self):
        self.pos = nx.spring_layout(self.G)
        self.ax.clear()

        for node, node_data in self.G.nodes(data=True):
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[node], 
                                   node_color=self.node_color_map[node_data['node_type']], 
                                   ax=self.ax)

        for edge_type in set(nx.get_edge_attributes(self.G, 'edge_type').values()):
            edges = [e for e, e_data in self.G.edges(data=True) if e_data['edge_type'] == edge_type]
            nx.draw_networkx_edges(self.G, self.pos, edgelist=edges, 
                                   edge_color=self.edge_color_map[edge_type], 
                                   ax=self.ax)

        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def run(self):
        seed_node_type = input("Enter seed node type: ")
        seed_node_id = input("Enter seed node ID: ")
        self.create_networkx_graph(seed_node_type, seed_node_id)
        self.draw_graph()

        while True:
            action = input("Enter 'expand <node_id>' to expand a node, or 'quit' to exit: ")
            if action.lower() == 'quit':
                break
            elif action.lower().startswith('expand'):
                _, node_id = action.split()
                node_type = self.G.nodes[node_id]['node_type']
                self.expand_graph(node_type, node_id)
                self.draw_graph()
            else:
                print("Invalid action. Please try again.")
