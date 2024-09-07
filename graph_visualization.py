import networkx as nx
import plotly.graph_objects as go

class GraphVisualizer:
    def __init__(self, node_color_map, edge_color_map):
        self.node_color_map = node_color_map
        self.edge_color_map = edge_color_map

    def create_interactive_subgraph(self, G, hetero_data):
        pos = nx.spring_layout(G)
        
        node_traces = {}
        for node_type, color in self.node_color_map.items():
            node_traces[node_type] = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                hoverinfo='text',
                marker=dict(color=color, size=10),
                name=node_type.capitalize(),
                customdata=[]
            )

        edge_traces = {}
        for edge_type in hetero_data.edge_types:
            src, rel, dst = edge_type
            edge_key = f"{src}_{dst}"
            edge_traces[edge_key] = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color=self.edge_color_map.get((src, dst), 'gray')),
                hoverinfo='none',
                mode='lines',
                name=f"{src.capitalize()}-{dst.capitalize()}"
            )

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_type = node[1]['type']
            node_traces[node_type]['x'] += (x,)
            node_traces[node_type]['y'] += (y,)
            node_traces[node_type]['text'] += (node[0],)
            node_traces[node_type]['customdata'] += (node[0],)

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_type = edge[2].get('type', ('unknown', 'unknown', 'unknown'))
            edge_key = f"{edge_type[0]}_{edge_type[2]}"
            if edge_key in edge_traces:
                edge_traces[edge_key]['x'] += (x0, x1, None)
                edge_traces[edge_key]['y'] += (y0, y1, None)

        fig = go.Figure(data=list(edge_traces.values()) + list(node_traces.values()))
        fig.update_layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        return fig