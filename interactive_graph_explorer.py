import networkx as nx
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import logging

from graph_visualization import GraphVisualizer
from graph_manager import GraphManager

class InteractiveGraphExplorer:
    def __init__(self, hetero_data, config, logger: logging.Logger):
        self.logger = logger
        self.config = config
        self.graph_manager = GraphManager(hetero_data, config, logger)
        self.visualizer = GraphVisualizer(config.node_color_map, self._process_edge_color_map())
        self.app = Dash(__name__)
        self.setup_layout()

    def _process_edge_color_map(self):
        return {
            k if isinstance(k, tuple) else tuple(k.split('_')): v 
            for k, v in self.config.edge_color_map.items()
        }

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Interactive Graph Explorer"),
            dcc.Dropdown(
                id='seed-node-type',
                options=[{'label': node_type.capitalize(), 'value': node_type} for node_type in self.graph_manager.node_types],
                value=self.graph_manager.node_types[0],
                placeholder="Select seed node type"
            ),
            dcc.Input(id='seed-node-input', type='text', placeholder='Enter seed node ID'),
            html.Button('Start Exploration', id='start-button', n_clicks=0),
            dcc.Graph(id='graph-visualization', figure=self.create_graph_figure()),
            html.Div(id='click-data')
        ])

        self.setup_callbacks()

    def setup_callbacks(self):
        @self.app.callback(
            Output('graph-visualization', 'figure'),
            [Input('start-button', 'n_clicks'),
             Input('graph-visualization', 'clickData')],
            [State('seed-node-type', 'value'),
             State('seed-node-input', 'value')]
        )
        def update_graph(n_clicks, clickData, seed_node_type, seed_node_id):
            ctx = callback_context
            if not ctx.triggered:
                return self.create_graph_figure()

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'start-button' and n_clicks > 0 and seed_node_id:
                self.graph_manager.create_graph(seed_node_type, seed_node_id)
            elif trigger_id == 'graph-visualization' and clickData:
                point = clickData['points'][0]
                if 'customdata' in point:
                    node_key = point['customdata']
                    node_type, node_id = node_key.split('_')
                    self.graph_manager.expand_graph(node_type, node_id)

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

    def create_graph_figure(self):
        return self.visualizer.create_interactive_subgraph(self.graph_manager.G, self.graph_manager.hetero_data)

    def run(self):
        self.app.run_server(debug=True)