import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import torch
from torch_geometric.data import HeteroData
import logging
from debug_info import DebugInfo

from graph_visualization import GraphVisualizer

# # Set up logging
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

class InteractiveGraphExplorer:
    def __init__(self, hetero_data: HeteroData, node_color_map: dict, edge_color_map: dict, logger: logging.Logger):
        self.logger = logger
        self.hetero_data = hetero_data
        self.node_color_map = node_color_map
        
        # Process edge_color_map to use tuples as keys
        self.edge_color_map = {
            k if isinstance(k, tuple) else tuple(k.split('_')): v 
            for k, v in edge_color_map.items()
        }
        
        self.G = nx.Graph()
        self.app = Dash(__name__)
        
        # Create reverse mappings for each node type
        self.reverse_mappings = {}
        for node_type in self.hetero_data.node_types:
            hashed_ids = self.hetero_data[node_type][f'{node_type}_id']
            original_ids = self.hetero_data[node_type].original_ids
            self.reverse_mappings[node_type] = dict(zip(hashed_ids.tolist(), original_ids))
        
        # Initialize GraphVisualizer
        self.visualizer = GraphVisualizer(self.node_color_map, self.edge_color_map)
        
        # Log debug information about nodes
        for node_type in self.hetero_data.node_types:
            node_ids = self.hetero_data[node_type][f'{node_type}_id']
            if len(node_ids) > 0:
                sample_id = node_ids[0].item()
                original_id = self.reverse_mappings[node_type][sample_id]
                logger.info(f"Sample {node_type} ID: {original_id} (hashed: {sample_id})")
                logger.info(f"Total number of {node_type}s: {len(node_ids)}")
            else:
                logger.warning(f"No {node_type}s found in the graph.")

        self.debug_info = DebugInfo()

        self.setup_layout()

    def get_node_id(self, node_type, index):
        hashed_id = self.hetero_data[node_type][f'{node_type}_id'][index].item()
        return self.reverse_mappings[node_type][hashed_id]

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Interactive Graph Explorer"),
            dcc.Dropdown(
                id='seed-node-type',
                options=[{'label': node_type.capitalize(), 'value': node_type} for node_type in self.hetero_data.node_types],
                value=self.hetero_data.node_types[0],
                placeholder="Select seed node type"
            ),
            dcc.Input(id='seed-node-input', type='text', placeholder='Enter seed node ID'),
            html.Button('Start Exploration', id='start-button', n_clicks=0),
            dcc.Graph(id='graph-visualization', figure=self.create_graph_figure()),
            html.Div(id='click-data')
        ])

        @self.app.callback(
            Output('graph-visualization', 'figure'),
            [Input('start-button', 'n_clicks'),
             Input('graph-visualization', 'clickData')],
            [State('seed-node-type', 'value'),
             State('seed-node-input', 'value')]
        )
        def update_graph(n_clicks, clickData, seed_node_type, seed_node_id):
            self.logger.info(f"Updating graph for {seed_node_type} node {seed_node_id}")
            ctx = callback_context
            if not ctx.triggered:
                return self.create_graph_figure()

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'start-button' and n_clicks > 0 and seed_node_id:
                self.create_networkx_graph(seed_node_type, seed_node_id)
            elif trigger_id == 'graph-visualization' and clickData:
                point = clickData['points'][0]
                if 'customdata' in point:
                    node_key = point['customdata']
                    node_type, node_id = node_key.split('_')
                    self.expand_graph(node_type, node_id)

            return self.create_graph_figure()

        @self.app.callback(
            Output('click-data', 'children'),
            [Input('graph-visualization', 'clickData')]
        )
        def display_click_data(clickData):
            if clickData is None:
                return "Click on a node to expand the graph"
            point = clickData['points'][0]
            return f"Clicked node: {point.get('customdata', 'Unknown')}"

    def create_networkx_graph(self, seed_node_type, seed_node_id):
        self.G.clear()
        self.logger.info(f"Creating NetworkX graph for {seed_node_type} node {seed_node_id}")

        hashed_seed_id = hash(seed_node_id)
        seed_index = (self.hetero_data[seed_node_type][f'{seed_node_type}_id'] == hashed_seed_id).nonzero(as_tuple=True)[0]
        if len(seed_index) == 0:
            self.logger.warning(f"Seed node ID {seed_node_id} not found in {seed_node_type}")
            return
        seed_index = seed_index[0].item()
        self.G.add_node(f"{seed_node_type}_{seed_node_id}", type=seed_node_type)

        for edge_type in self.hetero_data.edge_types:
            src, rel, dst = edge_type
            edge_index = self.hetero_data[edge_type].edge_index

            if src == seed_node_type:
                mask = edge_index[0] == seed_index
                connected_nodes = edge_index[1][mask]
                for node in connected_nodes:
                    hashed_dst_id = self.hetero_data[dst][f'{dst}_id'][node].item()
                    dst_id = self.reverse_mappings[dst][hashed_dst_id]
                    dst_node = f"{dst}_{dst_id}"
                    self.G.add_node(dst_node, type=dst)
                    self.G.add_edge(f"{seed_node_type}_{seed_node_id}", dst_node, type=edge_type)

            elif dst == seed_node_type:
                mask = edge_index[1] == seed_index
                connected_nodes = edge_index[0][mask]
                for node in connected_nodes:
                    hashed_src_id = self.hetero_data[src][f'{src}_id'][node].item()
                    src_id = self.reverse_mappings[src][hashed_src_id]
                    src_node = f"{src}_{src_id}"
                    self.G.add_node(src_node, type=src)
                    self.G.add_edge(src_node, f"{seed_node_type}_{seed_node_id}", type=edge_type)

    def create_graph_figure(self):
        return self.visualizer.create_interactive_subgraph(self.G, self.hetero_data)

    def expand_graph(self, node_type, node_id):
        node_key = f"{node_type}_{node_id}"
        if node_key not in self.G.nodes():
            return

        # Convert the node_id back to its hashed value
        hashed_node_id = hash(node_id)

        for edge_type in self.hetero_data.edge_types:
            src, rel, dst = edge_type
            edge_index = self.hetero_data[edge_type].edge_index

            if src == node_type:
                # Find the index of the node using the hashed ID
                src_mask = self.hetero_data[src][f'{src}_id'] == hashed_node_id
                src_idx = src_mask.nonzero(as_tuple=True)[0]
                if len(src_idx) > 0:
                    src_idx = src_idx[0]
                    connected_nodes = edge_index[1][edge_index[0] == src_idx]
                    for dst_idx in connected_nodes:
                        hashed_dst_id = self.hetero_data[dst][f'{dst}_id'][dst_idx].item()
                        dst_id = self.reverse_mappings[dst][hashed_dst_id]
                        dst_node = f"{dst}_{dst_id}"
                        self.G.add_node(dst_node, type=dst)
                        self.G.add_edge(node_key, dst_node, type=edge_type)

            elif dst == node_type:
                # Find the index of the node using the hashed ID
                dst_mask = self.hetero_data[dst][f'{dst}_id'] == hashed_node_id
                dst_idx = dst_mask.nonzero(as_tuple=True)[0]
                if len(dst_idx) > 0:
                    dst_idx = dst_idx[0]
                    connected_nodes = edge_index[0][edge_index[1] == dst_idx]
                    for src_idx in connected_nodes:
                        hashed_src_id = self.hetero_data[src][f'{src}_id'][src_idx].item()
                        src_id = self.reverse_mappings[src][hashed_src_id]
                        src_node = f"{src}_{src_id}"
                        self.G.add_node(src_node, type=src)
                        self.G.add_edge(src_node, node_key, type=edge_type)

    def run(self):
        self.app.run_server(debug=True)