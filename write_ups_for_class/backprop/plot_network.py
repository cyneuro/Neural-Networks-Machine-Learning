import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define node positions - spread them out more to avoid overlapping edges
pos = {
    'x1': (0, 1.5),
    'x2': (0, -1.5),
    'h1': (1.5, 1),
    'h2': (1.5, -1),
    'o': (3, 0)
}

# Add nodes
G.add_node('x1')
G.add_node('x2')
G.add_node('h1')
G.add_node('h2')
G.add_node('o')

# Add edges with weights
edges = [
    ('x1', 'h1', 0.2),
    ('x2', 'h1', 0.2),
    ('x1', 'h2', 0.3),
    ('x2', 'h2', 0.3),
    ('h1', 'o', 0.3),
    ('h2', 'o', 0.9)
]

for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# Node colors
node_colors = ['red', 'red', 'blue', 'blue', 'green']

# Draw the graph
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2500, font_size=16, font_weight='bold')

# Add edge labels
edge_labels = {(u, v): f'{w}' for u, v, w in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=12)

plt.title('Full Neural Network Architecture', fontsize=14, fontweight='bold')
plt.axis('off')
plt.savefig('network_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Mini-graph: Computing h1
# ============================================================================
G_h1 = nx.DiGraph()
pos_h1 = {'x1': (0, 1), 'x2': (0, 0), 'h1': (1, 0.5)}
G_h1.add_edge('x1', 'h1', weight=0.2)
G_h1.add_edge('x2', 'h1', weight=0.2)

plt.figure(figsize=(8, 5))
nx.draw(G_h1, pos_h1, with_labels=True, node_color=['red', 'red', 'blue'], 
        node_size=2000, font_size=14, font_weight='bold')
edge_labels_h1 = {('x1', 'h1'): '0.2', ('x2', 'h1'): '0.2'}
nx.draw_networkx_edge_labels(G_h1, pos_h1, edge_labels_h1, font_size=11)
plt.title('Computing Hidden Node 1 (h₁)', fontsize=12, fontweight='bold')
plt.axis('off')
plt.savefig('network_h1.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Mini-graph: Computing h2
# ============================================================================
G_h2 = nx.DiGraph()
pos_h2 = {'x1': (0, 1), 'x2': (0, 0), 'h2': (1, 0.5)}
G_h2.add_edge('x1', 'h2', weight=0.3)
G_h2.add_edge('x2', 'h2', weight=0.3)

plt.figure(figsize=(8, 5))
nx.draw(G_h2, pos_h2, with_labels=True, node_color=['red', 'red', 'blue'], 
        node_size=2000, font_size=14, font_weight='bold')
edge_labels_h2 = {('x1', 'h2'): '0.3', ('x2', 'h2'): '0.3'}
nx.draw_networkx_edge_labels(G_h2, pos_h2, edge_labels_h2, font_size=11)
plt.title('Computing Hidden Node 2 (h₂)', fontsize=12, fontweight='bold')
plt.axis('off')
plt.savefig('network_h2.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Mini-graph: Computing output
# ============================================================================
G_o = nx.DiGraph()
pos_o = {'h1': (0, 1), 'h2': (0, 0), 'o': (1, 0.5)}
G_o.add_edge('h1', 'o', weight=0.3)
G_o.add_edge('h2', 'o', weight=0.9)

plt.figure(figsize=(8, 5))
nx.draw(G_o, pos_o, with_labels=True, node_color=['blue', 'blue', 'green'], 
        node_size=2000, font_size=14, font_weight='bold')
edge_labels_o = {('h1', 'o'): '0.3', ('h2', 'o'): '0.9'}
nx.draw_networkx_edge_labels(G_o, pos_o, edge_labels_o, font_size=11)
plt.title('Computing Output Node (o)', fontsize=12, fontweight='bold')
plt.axis('off')
plt.savefig('network_output.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated network diagrams:")
print("  - network_diagram.png (full network)")
print("  - network_h1.png (h1 computation)")
print("  - network_h2.png (h2 computation)")
print("  - network_output.png (output computation)")