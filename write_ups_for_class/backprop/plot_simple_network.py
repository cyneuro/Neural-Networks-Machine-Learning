import matplotlib.pyplot as plt
import networkx as nx
import os

# Create a directed graph
G = nx.DiGraph()

# Define node positions
pos = {
    'x': (0, 0),
    'o': (1, 0)
}

# Add nodes
G.add_node('x')
G.add_node('o')

# Add edges with weights
G.add_edge('x', 'o', weight=0.8)

# Node colors
node_colors = ['red', 'green']

# Draw the graph
plt.figure(figsize=(8, 4))
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=3000, font_size=20, font_weight='bold')

# Add edge labels
edge_labels = {('x', 'o'): 'w = 0.8'}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=16)

plt.title('Simple 1-to-1 Neural Network Architecture', fontsize=18, fontweight='bold')
plt.axis('off')

# Save to the same directory as the script
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simple_network.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved figure to {save_path}")
