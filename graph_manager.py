import networkx as nx
import torch
from torch_geometric.data import HeteroData
import logging

class GraphManager:
    def __init__(self, hetero_data: HeteroData, config, logger: logging.Logger):
        self.logger = logger
        self.hetero_data = hetero_data
        self.config = config
        self.node_types = list(config.node_types.keys())
        self.G = nx.Graph()
        self.reverse_mappings = self._create_reverse_mappings()

    def _create_reverse_mappings(self):
        return {
            node_type: dict(zip(self.hetero_data[node_type][self.config.node_types[node_type]].tolist(), 
                                self.hetero_data[node_type].original_ids))
            for node_type in self.hetero_data.node_types
        }

    def create_graph(self, seed_node_type, seed_node_id):
        self.G.clear()
        self.logger.info(f"Creating graph for {seed_node_type} node {seed_node_id}")

        hashed_seed_id = hash(seed_node_id)
        seed_index = (self.hetero_data[seed_node_type][self.config.node_types[seed_node_type]] == hashed_seed_id).nonzero(as_tuple=True)[0]
        if len(seed_index) == 0:
            self.logger.warning(f"Seed node ID {seed_node_id} not found in {seed_node_type}")
            return
        seed_index = seed_index[0].item()
        self.G.add_node(f"{seed_node_type}_{seed_node_id}", type=seed_node_type)

        self._add_connected_nodes(seed_node_type, seed_index, seed_node_id)

    def _add_connected_nodes(self, node_type, node_index, node_id):
        for edge_type in self.hetero_data.edge_types:
            src, rel, dst = edge_type
            edge_index = self.hetero_data[edge_type].edge_index

            if src == node_type:
                self._add_outgoing_edges(edge_index, node_index, src, dst, node_id)
            elif dst == node_type:
                self._add_incoming_edges(edge_index, node_index, src, dst, node_id)

    def _add_outgoing_edges(self, edge_index, src_index, src_type, dst_type, src_id):
        mask = edge_index[0] == src_index
        connected_nodes = edge_index[1][mask]
        for node in connected_nodes:
            hashed_dst_id = self.hetero_data[dst_type][self.config.node_types[dst_type]][node].item()
            dst_id = self.reverse_mappings[dst_type][hashed_dst_id]
            dst_node = f"{dst_type}_{dst_id}"
            self.G.add_node(dst_node, type=dst_type)
            self.G.add_edge(f"{src_type}_{src_id}", dst_node, type=(src_type, dst_type))  # Store as tuple

    def _add_incoming_edges(self, edge_index, dst_index, src_type, dst_type, dst_id):
        mask = edge_index[1] == dst_index
        connected_nodes = edge_index[0][mask]
        for node in connected_nodes:
            hashed_src_id = self.hetero_data[src_type][self.config.node_types[src_type]][node].item()
            src_id = self.reverse_mappings[src_type][hashed_src_id]
            src_node = f"{src_type}_{src_id}"
            self.G.add_node(src_node, type=src_type)
            self.G.add_edge(src_node, f"{dst_type}_{dst_id}", type=(src_type, dst_type))  # Store as tuple

    def expand_graph(self, node_type, node_id):
        node_key = f"{node_type}_{node_id}"
        if node_key not in self.G.nodes():
            return

        hashed_node_id = hash(node_id)
        node_index = (self.hetero_data[node_type][self.config.node_types[node_type]] == hashed_node_id).nonzero(as_tuple=True)[0]
        if len(node_index) == 0:
            self.logger.warning(f"Node ID {node_id} not found in {node_type}")
            return
        node_index = node_index[0].item()

        self._add_connected_nodes(node_type, node_index, node_id)

    def get_node_id(self, node_type, index):
        hashed_id = self.hetero_data[node_type][self.config.node_types[node_type]][index].item()
        return self.reverse_mappings[node_type][hashed_id]