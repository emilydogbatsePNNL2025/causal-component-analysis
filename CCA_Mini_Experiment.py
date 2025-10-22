# CCA_Mini_Experiment


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
pl.seed_everything(42)

print("Successfully imported all libraries and set random seeds.")

# Step 1: Defining a simple Causal graph 

"""
This part creates a simple 3 variable causal chain : Z1 -> Z2 -> Z3
"""


# 1 means there's a causal edge from i to j
adjacency = np.array([
    [0, 1, 0],  # Z1 causes Z2
    [0, 0, 1],  # Z2 causes Z3
    [0, 0, 0]   # Z3 has no children
])

print("Causal Structure:")
print("Z1 → Z2 → Z3")
print("\nAdjacency Matrix:")
print(adjacency)

# visualizing the graph now to be able to visualize how the causal structure looks like
# Visualize the graph
fig, ax = plt.subplots(figsize=(8, 3))
ax.text(0.1, 0.5, 'Z1', fontsize=20, ha='center', va='center',
        bbox=dict(boxstyle='circle', facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(0.5, 0.5, 'Z2', fontsize=20, ha='center', va='center',
        bbox=dict(boxstyle='circle', facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(0.9, 0.5, 'Z3', fontsize=20, ha='center', va='center',
        bbox=dict(boxstyle='circle', facecolor='lightblue', edgecolor='black', linewidth=2))

# Draw arrows
ax.annotate('', xy=(0.45, 0.5), xytext=(0.15, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(0.85, 0.5), xytext=(0.55, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Causal Graph Structure', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./outputs/01_causal_graph.png', dpi=150, bbox_inches='tight')
plt.show()

print("Causal graph defined")
