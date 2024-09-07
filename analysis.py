import torch
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from typing import Any, Dict, List, Tuple, Callable
import os
import signal

from graph_visualization import GraphVisualizer
from core.graph_builder import create_heterogeneous_graph

def run_analysis(data_path: str, node_types: Dict[str, str], edge_types: List[Tuple[str, str, str]], 
                 node_color_map: Dict[str, str], edge_color_map: Dict[Tuple[str, str], str], logger) -> Dict[str, Any]:
    results = {}
    
    # Create heterogeneous graph
    graph = create_heterogeneous_graph(data_path, node_types, edge_types, logger)
    
    # Create interactive visualization
    visualizer = GraphVisualizer(node_color_map, edge_color_map)
    interactive_fig = visualizer.create_interactive_subgraph(graph, graph)
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
        ctx = dash.callback_context
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
        
        return visualizer.create_interactive_subgraph(graph, graph)
    
    return results, app