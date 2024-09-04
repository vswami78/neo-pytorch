import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Callable
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import os
import signal
import json
from typing import Dict, List, Any
from interactive_graph_explorer import InteractiveGraphExplorer
import sys

# Set up logging
log_file = 'graph_explorer.log'
logging.basicConfig(level=logging.ERROR,  # Change this to control overall logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        RotatingFileHandler(log_file, maxBytes=1000000, backupCount=5),
                        logging.StreamHandler(sys.stderr)
                    ])

# # Silence other loggers
# logging.getLogger("matplotlib").setLevel(logging.WARNING)
# logging.getLogger("networkx").setLevel(logging.WARNING)
# logging.getLogger("torch").setLevel(logging.WARNING)
# logging.getLogger("dash").setLevel(logging.WARNING)

# Create a logger for your application
logger = logging.getLogger("graph_explorer")
logger.setLevel(logging.ERROR)  # Set this to control your application's logging level

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

def create_heterogeneous_graph(data_path, node_types, edge_types):
    """
    Create a heterogeneous graph from tabular data.
    
    :param data_path: Path to the CSV file containing the tabular data
    :param node_types: Dictionary mapping node types to their identifying columns
    :param edge_types: List of tuples defining edge types (source, edge_type, target)
    :return: PyTorch Geometric HeteroData object
    """
    if isinstance(data_path, HeteroData):
        return data_path  # Return the HeteroData object directly
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns}")
    data = HeteroData()
    
    # Create nodes
    for node_type, id_col in node_types.items():
        unique_ids = np.unique(df[id_col].to_numpy())
        data[node_type].x = torch.arange(len(unique_ids))
        data[node_type].mapping = {id: i for i, id in enumerate(unique_ids)}
        data[node_type][f'{node_type}_id'] = unique_ids
        logger.info(f"Created {len(unique_ids)} {node_type} nodes")
        logger.debug(f"Sample {node_type} IDs: {unique_ids[:5]}")
    
    # Create edges
    logger.info(f"Edge types to create: {edge_types}")
    for src_type, edge_type, dst_type in edge_types:
        logger.info(f"Creating edges for {src_type}-{edge_type}-{dst_type}")
        src_col = node_types[src_type]
        dst_col = node_types[dst_type]
        logger.debug(f"Source column: {src_col}, Destination column: {dst_col}")
        
        src_ids = df[src_col].map(lambda x: data[src_type].mapping.get(x, -1))
        dst_ids = df[dst_col].map(lambda x: data[dst_type].mapping.get(x, -1))
        
        mask = (src_ids != -1) & (dst_ids != -1)
        src_ids = src_ids[mask]
        dst_ids = dst_ids[mask]
        
        logger.info(f"Number of valid edges: {len(src_ids)}")
        
        if len(src_ids) > 0:
            edge_index = torch.stack([torch.tensor(src_ids.values), torch.tensor(dst_ids.values)], dim=0)
            data[src_type, edge_type, dst_type].edge_index = edge_index
            logger.info(f"Added {edge_index.shape[1]} edges for {src_type}-{edge_type}-{dst_type}")
        else:
            logger.warning(f"No valid edges found for {src_type}-{edge_type}-{dst_type}")
        
        if src_type == 'user' and dst_type == 'product':
            logger.debug("Checking for U1010-P85 relationship")
            u1010_index = data['user'].mapping.get('U1010', -1)
            p85_index = data['product'].mapping.get('P85', -1)
            logger.debug(f"U1010 index: {u1010_index}, P85 index: {p85_index}")
            if u1010_index != -1 and p85_index != -1:
                if len(src_ids) > 0:
                    mask = (edge_index[0] == u1010_index) & (edge_index[1] == p85_index)
                    if mask.any():
                        logger.info("Relationship between U1010 and P85 found in the graph")
                    else:
                        logger.info("Relationship between U1010 and P85 NOT found in the graph")
                else:
                    logger.info("No edges to check for U1010-P85 relationship")
            else:
                logger.info("U1010 or P85 not found in mappings")
    
    logger.info(f"Final graph node types: {data.node_types}")
    logger.info(f"Final graph edge types: {data.edge_types}")
    return data

def create_interactive_subgraph(graph, filters, node_color_map, edge_color_map):
    """
    Create an interactive subgraph visualization using Plotly.
    
    :param graph: PyTorch Geometric HeteroData object
    :param filters: Dictionary of node types and their filter functions
    :param node_color_map: Dictionary mapping node types to colors
    :param edge_color_map: Dictionary mapping edge types to colors
    :return: Plotly Figure object
    """
    G = nx.Graph()
    
    # Apply filters and add nodes
    for node_type in graph.node_types:
        filter_funcs = filters.get(node_type, [lambda x: torch.ones(x.num_nodes, dtype=torch.bool)])
        mask = torch.ones(graph[node_type].num_nodes, dtype=torch.bool)
        for func in filter_funcs:
            mask &= func(graph[node_type])
        
        for i in range(graph[node_type].num_nodes):
            if mask[i]:
                node_id = graph[node_type][f'{node_type}_id'][i]
                if hasattr(node_id, 'item'):
                    node_id = node_id.item()
                G.add_node(f"{node_type}_{node_id}", type=node_type)
    
    # Add edges
    for edge_type in graph.edge_types:
        src, _, dst = edge_type
        edge_index = graph[edge_type].edge_index.t().tolist()
        for src_idx, dst_idx in edge_index:
            src_id_value = graph[src][f'{src}_id'][src_idx]
            src_id = src_id_value.item() if hasattr(src_id_value, 'item') else src_id_value
            dst_id_value = graph[dst][f'{dst}_id'][dst_idx]
            dst_id = dst_id_value.item() if hasattr(dst_id_value, 'item') else dst_id_value
            src_node = f"{src}_{src_id}"
            dst_node = f"{dst}_{dst_id}"
            if G.has_node(src_node) and G.has_node(dst_node):
                G.add_edge(src_node, dst_node)
    
    # Create layout
    pos = nx.spring_layout(G)
    
    # Create traces for nodes
    node_traces = {}
    for node_type in node_color_map:
        node_traces[node_type] = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=node_color_map[node_type],
                size=10,
            ),
            name=node_type.capitalize()
        )
    
    # Create traces for edges
    edge_traces = {}
    for edge_type in edge_color_map:
        edge_traces[edge_type] = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color=edge_color_map[edge_type]),
            hoverinfo='none',
            mode='lines',
            name=f"{edge_type[0].capitalize()}-{edge_type[1].capitalize()}"
        )
    
    # Add node and edge positions to traces
    for node in G.nodes():
        x, y = pos[node]
        node_type = G.nodes[node]['type']
        node_traces[node_type]['x'] += (x,)
        node_traces[node_type]['y'] += (y,)
        node_traces[node_type]['text'] += (node,)
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_type = (G.nodes[edge[0]]['type'], G.nodes[edge[1]]['type'])
        edge_traces[edge_type]['x'] += (x0, x1, None)
        edge_traces[edge_type]['y'] += (y0, y1, None)
    
    # Create figure
    fig = go.Figure(data=list(edge_traces.values()) + list(node_traces.values()))
    fig.update_layout(
        title="Interactive Subgraph Visualization",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[dict(
            text="",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def run_analysis(data_path: str, node_types: Dict[str, str], edge_types: List[Tuple[str, str, str]], 
                 node_color_map: Dict[str, str], edge_color_map: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
    results = {}
    
    # Create heterogeneous graph
    graph = create_heterogeneous_graph(data_path, node_types, edge_types)
    
    # Create interactive visualization
    initial_filters = {
        node_type: [lambda x: torch.ones(x.num_nodes, dtype=torch.bool)]
        for node_type in graph.node_types
    }
    interactive_fig = create_interactive_subgraph(graph, initial_filters, node_color_map, edge_color_map)
    results['interactive_visualization'] = interactive_fig
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        dcc.Graph(id='subgraph-plot', figure=interactive_fig),
        html.Div([
            html.Label('Edge Count Filter:'),
            dcc.Slider(
                id='edge-count-filter',
                min=0,
                max=10,
                step=1,
                value=0,
                marks={i: str(i) for i in range(11)}
            )
        ]),
        html.Button('Quit Application', id='quit-button', n_clicks=0)
    ])
    
    @app.callback(
        Output('subgraph-plot', 'figure'),
        [Input('edge-count-filter', 'value'),
         Input('quit-button', 'n_clicks')]
    )
    def update_graph(edge_count, n_clicks):
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'quit-button.n_clicks':
            os.kill(os.getpid(), signal.SIGINT)
            return dash.no_update

        filters = {
            node_type: [lambda x: torch.ones(x.num_nodes, dtype=torch.bool)]
            for node_type in graph.node_types
        }
        
        # Get the first node type and edge type from the config
        primary_node_type = next(iter(node_types.keys()))
        primary_edge_type = edge_types[0]

        if edge_count > 0:
            filters[primary_node_type].append(lambda x: torch.bincount(graph[primary_edge_type].edge_index[0]) > edge_count)
        
        return create_interactive_subgraph(graph, filters, node_color_map, edge_color_map)
    
    return results, app

def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load and parse the configuration file.
        
        :param config_path: Path to the JSON configuration file
        :return: Dictionary containing the parsed configuration
        """
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
            
            required_keys = ['input_file', 'node_types', 'edge_types', 'node_color_map', 'edge_color_map']
            for key in required_keys:
                if key not in config:
                    raise KeyError(f"Missing required key in config: {key}")
            
            # Convert edge_color_map keys to tuples
            config['edge_color_map'] = {tuple(k.split('_')): v for k, v in config['edge_color_map'].items()}
            
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
        except KeyError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            raise

def validate_config(config: Dict[str, Any], df: pd.DataFrame) -> None:
    """
    Validate the configuration against the loaded data.
    
    :param config: Parsed configuration dictionary
    :param df: Loaded DataFrame
    """
    # Collect all column names from node_types
    all_columns = set()
    for columns in config['node_types'].values():
        if isinstance(columns, str):
            all_columns.add(columns)
        elif isinstance(columns, list):
            all_columns.update(columns)
    
    # Check if all specified columns exist in the DataFrame
    missing_columns = all_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns specified in config not found in data: {missing_columns}")
    
    # Validate edge types
    node_types = set(config['node_types'].keys())
    for edge in config['edge_types']:
        if len(edge) != 3:
            raise ValueError(f"Invalid edge specification: {edge}. Should be [source, edge_type, target]")
        if edge[0] not in node_types or edge[2] not in node_types:
            raise ValueError(f"Invalid node type in edge specification: {edge}")

# Usage

if __name__ == "__main__":
    config_path = "graph_config.json"
    config = load_config(config_path)
    
    data_path = config['input_file']
    node_types = config['node_types']
    edge_types = [tuple(edge) for edge in config['edge_types'] if len(edge) == 3]
    if len(edge_types) != len(config['edge_types']):
        logger.warning("Some edge types were skipped because they didn't have exactly 3 elements.")

    # Use color maps from config
    node_color_map = config['node_color_map']
    edge_color_map = config['edge_color_map']

    # Validate config against the data
    df = pd.read_csv(data_path)
    validate_config(config, df)
    
    # Create heterogeneous graph
    hetero_data = create_heterogeneous_graph(data_path, node_types, edge_types)
    
    # Create and run the InteractiveGraphExplorer
    explorer = InteractiveGraphExplorer(hetero_data, node_color_map, edge_color_map)
    explorer.run()
