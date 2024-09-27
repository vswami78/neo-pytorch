import networkx as nx
import plotly.graph_objects as go

class GraphVisualizer:
    def __init__(self, node_color_map, edge_color_map, edge_width_map):
        self.node_color_map = node_color_map
        self.edge_color_map = edge_color_map
        self.edge_width_map = edge_width_map  # Add this line

    def create_interactive_subgraph(self, G, hetero_data):
        if len(G.edges()) == 0:
            return self._create_empty_figure()

        pos = nx.spring_layout(G)
        edge_traces = self._create_edge_traces(G, pos, hetero_data)
        node_traces = self._create_node_traces(G, pos)
        
        scatter_traces = self._create_scatter_traces(edge_traces, node_traces)
        
        return go.Figure(data=scatter_traces, layout=self._create_layout())

    def _create_empty_figure(self):
        return go.Figure(layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="No data to display yet. Please interact to populate the graph.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=14)
                )
            ]
        ))

    def _create_edge_traces(self, G, pos, hetero_data):
        edge_traces = {}
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_type = edge[2]['type']
            if edge_type not in edge_traces:
                edge_traces[edge_type] = {
                    'x': [],
                    'y': [],
                    'color': self.edge_color_map.get(edge_type, '#888'),
                    'width': self.edge_width_map.get(edge_type, 0.5),  # Add this line
                    'name': f"{edge_type[0].capitalize()}-{edge_type[1].capitalize()}" if isinstance(edge_type, tuple) else str(edge_type)
                }
            edge_traces[edge_type]['x'].extend([x0, x1, None])
            edge_traces[edge_type]['y'].extend([y0, y1, None])
        return edge_traces

    def _create_node_traces(self, G, pos):
        node_traces = {}
        for node, node_data in G.nodes(data=True):
            x, y = pos[node]
            node_type = node_data['type']
            if node_type not in node_traces:
                node_traces[node_type] = {
                    'x': [],
                    'y': [],
                    'text': [],
                    'customdata': []
                }
            node_traces[node_type]['x'].append(x)
            node_traces[node_type]['y'].append(y)
            node_traces[node_type]['text'].append(f"{node_type}: {node}")
            node_traces[node_type]['customdata'].append(node)
        return node_traces

    def _create_scatter_traces(self, edge_traces, node_traces):
        scatter_traces = []
        for edge_type, edge_data in edge_traces.items():
            scatter_traces.append(go.Scatter(
                x=edge_data['x'],
                y=edge_data['y'],
                line=dict(width=edge_data['width'], color=edge_data['color']),  # Modified this line
                hoverinfo='none',
                mode='lines',
                name=edge_data['name']
            ))
        for node_type, node_data in node_traces.items():
            scatter_traces.append(go.Scatter(
                x=node_data['x'],
                y=node_data['y'],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    color=self.node_color_map[node_type],
                    size=10,
                    line_width=2
                ),
                text=node_data['text'],
                customdata=node_data['customdata'],
                name=node_type.capitalize()
            ))
        return scatter_traces

    def _create_layout(self):
        return go.Layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )