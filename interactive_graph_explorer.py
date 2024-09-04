import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import torch
from torch_geometric.data import HeteroData
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveGraphExplorer:
    def __init__(self, hetero_data: HeteroData, node_color_map: dict, edge_color_map: dict):
        self.hetero_data = hetero_data
        self.node_color_map = node_color_map
        self.edge_color_map = edge_color_map
        self.G = nx.Graph()
        self.app = Dash(__name__)
        self.setup_layout()

        # Log debug information about users
        if 'user' in self.hetero_data.node_types:
            user_ids = self.hetero_data['user'].user_id
            if len(user_ids) > 0:
                sample_user_id = user_ids[0].item() if torch.is_tensor(user_ids[0]) else user_ids[0]
                logger.info(f"Sample user ID: {sample_user_id}")
                logger.info(f"Total number of users: {len(user_ids)}")
            else:
                logger.warning("No users found in the graph.")
        else:
            logger.warning("No 'user' node type found in the graph.")
        
        self.app.layout = html.Div([
            html.H1("Interactive Graph Explorer"),
            dcc.Input(id='seed-node-input', type='text', placeholder='Enter seed node ID'),
            html.Button('Start Exploration', id='start-button', n_clicks=0),
            dcc.Graph(id='graph-visualization'),
            html.Div(id='click-data', style={'whiteSpace': 'pre-wrap'})
        ])

        @self.app.callback(
            [Output('graph-visualization', 'figure'),
             Output('click-data', 'children')],
            [Input('start-button', 'n_clicks'),
             Input('graph-visualization', 'clickData')],
            [State('seed-node-input', 'value')]
        )
        def update_graph(n_clicks, clickData, seed_node):
            ctx = callback_context
            if not ctx.triggered:
                return go.Figure(), "Enter a seed node ID and click 'Start Exploration' to begin."

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'start-button':
                if not seed_node:
                    return go.Figure(), "Please enter a seed node ID."
                self.G = nx.Graph()
                self.add_node_and_neighbors(seed_node, 'user')
            elif trigger_id == 'graph-visualization' and clickData:
                clicked_node = self.extract_clicked_node(clickData)
                if clicked_node and clicked_node in self.G.nodes:
                    node_type = self.G.nodes[clicked_node]['type']
                    self.add_node_and_neighbors(clicked_node, node_type)

            return self.create_graph_figure(), f"Clicked node: {clicked_node if 'clicked_node' in locals() else 'None'}"

    def setup_layout(self):
        # Define your layout setup logic here
        pass  # Remove this 'pass' and add your actual layout setup code

    def add_node_to_graph(self, node_id, node_type):
        if node_id not in self.G.nodes:
            self.G.add_node(node_id, type=node_type)

    def get_user_products(self, user_id):
        edge_index = self.hetero_data['user', 'purchased', 'product'].edge_index
        user_index = self.hetero_data['user'].mapping.get(user_id, -1)
        if user_index != -1:
            product_indices = edge_index[1][edge_index[0] == user_index]
            return [self.hetero_data['product'].product_id[i] for i in product_indices]
        return []

    def get_product_users(self, product_id):
        edge_index = self.hetero_data['user', 'purchased', 'product'].edge_index
        product_index = self.hetero_data['product'].mapping.get(product_id, -1)
        if product_index != -1:
            user_indices = edge_index[0][edge_index[1] == product_index]
            return [self.hetero_data['user'].user_id[i] for i in user_indices]
        return []

    def create_graph_figure(self):
        pos = nx.spring_layout(self.G)
        
        edge_traces = []
        for (start_type, end_type), color in self.edge_color_map.items():
            logger.debug(f"Drawing edges for {start_type}-{end_type} with color {color}")
            edge_x, edge_y = [], []
            for edge in self.G.edges():
                if (self.G.nodes[edge[0]]['type'] == start_type and self.G.nodes[edge[1]]['type'] == end_type) or \
                   (self.G.nodes[edge[1]]['type'] == start_type and self.G.nodes[edge[0]]['type'] == end_type):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            edge_traces.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color=color),
                hoverinfo='none',
                mode='lines'))

        # Create node traces
        node_traces = []
        for node_type, color in self.node_color_map.items():
            node_x, node_y = [], []
            node_text = []
            for node, attrs in self.G.nodes(data=True):
                if attrs['type'] == node_type:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"{node} ({attrs['type']})")
            node_traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    color=color,
                    size=10,
                    line_width=2),
                text=node_text,
                hovertext=node_text,
                name=node_type.capitalize()))

        # Combine all traces
        traces = edge_traces + node_traces

        fig = go.Figure(data=traces,
                        layout=go.Layout(
                            title='Interactive Graph Visualization',
                            titlefont_size=16,
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 ) ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        return fig

    def run(self):
        self.app.run_server(debug=True)

    def add_node_and_neighbors(self, node_id, node_type):
        self.add_node_to_graph(node_id, node_type)
        if node_type == 'user':
            products = self.get_user_products(node_id)
            for product in products:
                self.add_node_to_graph(product, 'product')
                self.G.add_edge(node_id, product)
                category = self.get_product_category(product)
                if category:
                    self.add_node_to_graph(category, 'category')
                    self.G.add_edge(product, category)
        elif node_type == 'product':
            users = self.get_product_users(node_id)
            for user in users:
                self.add_node_to_graph(user, 'user')
                self.G.add_edge(node_id, user)
            category = self.get_product_category(node_id)
            if category:
                self.add_node_to_graph(category, 'category')
                self.G.add_edge(node_id, category)
        elif node_type == 'category':
            products = self.get_category_products(node_id)
            for product in products:
                self.add_node_to_graph(product, 'product')
                self.G.add_edge(node_id, product)
                users = self.get_product_users(product)
                for user in users:
                    self.add_node_to_graph(user, 'user')
                    self.G.add_edge(product, user)

    def extract_clicked_node(self, clickData):
        if 'points' in clickData and clickData['points']:
            point = clickData['points'][0]
            if 'text' in point:
                return point['text'].split(' ')[0]
            elif 'label' in point:
                return point['label'].split(' ')[0]
            elif 'customdata' in point:
                return point['customdata']
        return None

    def get_product_category(self, product_id):
        edge_index = self.hetero_data['product', 'belongs_to', 'category'].edge_index
        product_index = self.hetero_data['product'].mapping.get(product_id, -1)
        if product_index != -1:
            category_indices = edge_index[1][edge_index[0] == product_index]
            if len(category_indices) > 0:
                category_id = self.hetero_data['category'].category_id[category_indices[0]]
                return category_id if isinstance(category_id, str) else category_id.item()
        return None

    def get_category_products(self, category_id):
        edge_index = self.hetero_data['product', 'belongs_to', 'category'].edge_index
        category_index = self.hetero_data['category'].mapping.get(category_id, -1)
        if category_index != -1:
            product_indices = edge_index[0][edge_index[1] == category_index]
            return [self.hetero_data['product'].product_id[i] for i in product_indices]
        return []

# # Usage
# if __name__ == "__main__":
#     # Assume hetero_data is your HeteroData object
#     explorer = InteractiveGraphExplorer(hetero_data)
#     explorer.run()