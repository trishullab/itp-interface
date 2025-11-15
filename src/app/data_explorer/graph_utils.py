"""
Graph analysis utilities for dependency visualization and forest detection.
"""

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import networkx as nx
import plotly.graph_objects as go
from typing import List, Set, Dict, Tuple, Any
from itp_interface.tools.simple_sqlite import LeanDeclarationDB


def build_dependency_graph(db: LeanDeclarationDB) -> nx.DiGraph:
    """
    Build a directed graph from the dependency relationships.

    Args:
        db: LeanDeclarationDB instance

    Returns:
        NetworkX directed graph where edges point from dependent to dependency
        (A -> B means A depends on B)
    """
    G = nx.DiGraph()

    # Add all declarations as nodes
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT decl_id, name, namespace, decl_type, file_path
        FROM declarations
    """)

    for row in cursor.fetchall():
        decl_id, name, namespace, decl_type, file_path = row
        full_name = f"{namespace}.{name}" if namespace else name
        G.add_node(decl_id, name=name, full_name=full_name,
                   namespace=namespace, decl_type=decl_type,
                   file_path=file_path)

    # Add all dependency edges
    cursor.execute("""
        SELECT from_decl_id, to_decl_id
        FROM declaration_dependencies
    """)

    for from_id, to_id in cursor.fetchall():
        G.add_edge(from_id, to_id)

    return G


def find_all_forests(G: nx.DiGraph) -> List[Set[str]]:
    """
    Find all connected components (forests) in the graph.

    Args:
        G: NetworkX directed graph

    Returns:
        List of sets, where each set contains decl_ids in a connected component
    """
    # Convert to undirected for finding connected components
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))

    # Sort by size (largest first)
    components.sort(key=len, reverse=True)

    return components


def get_dependencies_closure(
    db: LeanDeclarationDB,
    decl_id: str,
    max_depth: int = 10
) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
    """
    Get all dependencies (transitive) of a declaration.

    Args:
        db: LeanDeclarationDB instance
        decl_id: Starting declaration ID
        max_depth: Maximum depth to traverse

    Returns:
        Tuple of (list of declaration dicts, subgraph)
    """
    G = build_dependency_graph(db)

    if decl_id not in G:
        return [], nx.DiGraph()

    # Get all reachable nodes (BFS limited by depth)
    visited = set()
    queue = [(decl_id, 0)]
    visited.add(decl_id)

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for neighbor in G.successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    # Create subgraph
    subgraph = G.subgraph(visited).copy()

    # Get declaration info for all nodes
    decls = []
    for node_id in visited:
        decl = db.get_declaration_by_decl_id(node_id)
        if decl:
            decls.append(decl)

    return decls, subgraph


def get_dependents_closure(
    db: LeanDeclarationDB,
    decl_id: str,
    max_depth: int = 10
) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
    """
    Get all dependents (transitive) of a declaration.

    Args:
        db: LeanDeclarationDB instance
        decl_id: Starting declaration ID
        max_depth: Maximum depth to traverse

    Returns:
        Tuple of (list of declaration dicts, subgraph)
    """
    G = build_dependency_graph(db)

    if decl_id not in G:
        return [], nx.DiGraph()

    # Get all nodes that can reach this node (reverse BFS)
    visited = set()
    queue = [(decl_id, 0)]
    visited.add(decl_id)

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for neighbor in G.predecessors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    # Create subgraph
    subgraph = G.subgraph(visited).copy()

    # Get declaration info for all nodes
    decls = []
    for node_id in visited:
        decl = db.get_declaration_by_decl_id(node_id)
        if decl:
            decls.append(decl)

    return decls, subgraph


def visualize_graph(G: nx.DiGraph, title: str = "Dependency Graph") -> go.Figure:
    """
    Create an interactive Plotly visualization of the graph.

    Args:
        G: NetworkX directed graph
        title: Title for the plot

    Returns:
        Plotly Figure object
    """
    if len(G.nodes()) == 0:
        # Empty graph
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[
            dict(text="No dependencies found", showarrow=False,
                 xref="paper", yref="paper", x=0.5, y=0.5)
        ])
        return fig

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    # Color map for declaration types
    type_colors = {
        'theorem': '#FF6B6B',
        'def': '#4ECDC4',
        'axiom': '#45B7D1',
        'instance': '#FFA07A',
        'class': '#98D8C8',
        None: '#95A5A6'
    }

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Get node info
        node_data = G.nodes[node]
        name = node_data.get('full_name', node_data.get('name', 'Unknown'))
        decl_type = node_data.get('decl_type')
        file_path = node_data.get('file_path', 'Unknown')

        node_text.append(f"{name}<br>Type: {decl_type}<br>File: {file_path}")
        node_color.append(type_colors.get(decl_type, type_colors[None]))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2))

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig


def get_forest_statistics(G: nx.DiGraph, forest: Set[str]) -> Dict[str, Any]:
    """
    Get statistics for a forest (connected component).

    Args:
        G: NetworkX directed graph
        forest: Set of decl_ids in the forest

    Returns:
        Dictionary with forest statistics
    """
    subgraph = G.subgraph(forest)

    # Find root nodes (nodes with no incoming edges from within forest)
    roots = [node for node in forest if subgraph.in_degree(node) == 0]

    # Find leaf nodes (nodes with no outgoing edges from within forest)
    leaves = [node for node in forest if subgraph.out_degree(node) == 0]

    return {
        'size': len(forest),
        'num_edges': subgraph.number_of_edges(),
        'num_roots': len(roots),
        'num_leaves': len(leaves),
        'roots': [G.nodes[r].get('full_name', r) for r in roots[:5]],  # First 5 roots
        'leaves': [G.nodes[l].get('full_name', l) for l in leaves[:5]]  # First 5 leaves
    }
