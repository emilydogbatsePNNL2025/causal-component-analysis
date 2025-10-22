# %% [markdown]
# # CauCA Mini-Experiment
# 
# This notebook demonstrates the full CauCA pipeline on simple synthetic data:
# 1. Generate causal latent variables
# 2. Mix them nonlinearly into observations
# 3. Train CauCA to recover the latents
# 4. Visualize and evaluate recovery quality

# %%
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

print("✓ Imports successful")

# %% [markdown]
# ## Step 1: Define a Simple Causal Graph
# 
# We'll create a 3-variable causal chain: Z1 → Z2 → Z3

# %%
# Define the causal adjacency matrix
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

print("✓ Causal graph defined")

# %% [markdown]
# ## Step 2: Generate Latent Causal Variables
# 
# Generate data following the causal structure with interventions

# %%
def generate_causal_data(n_samples=1000, n_environments=3):
    """Generate latent causal variables following the graph structure"""
    
    all_latents = []
    all_envs = []
    
    for env in range(n_environments):
        latents = np.zeros((n_samples, 3))
        
        # Z1: Root variable (different mean per environment)
        latents[:, 0] = np.random.normal(loc=env*0.5, scale=1.0, size=n_samples)
        
        # Z2: Caused by Z1
        latents[:, 1] = 0.8 * latents[:, 0] + np.random.normal(0, 0.5, n_samples)
        
        # Z3: Caused by Z2
        latents[:, 2] = 0.7 * latents[:, 1] + np.random.normal(0, 0.5, n_samples)
        
        all_latents.append(latents)
        all_envs.append(np.ones(n_samples) * env)
    
    latents = np.vstack(all_latents)
    envs = np.hstack(all_envs)
    
    return latents, envs

# Generate data
n_samples_per_env = 500
latents_true, environments = generate_causal_data(n_samples=n_samples_per_env, n_environments=3)

print(f"Generated {len(latents_true)} samples across {len(np.unique(environments))} environments")
print(f"Latent variables shape: {latents_true.shape}")

# Visualize the latent variables
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for i in range(3):
    for env in range(3):
        mask = environments == env
        axes[i].plot(np.where(mask)[0], latents_true[mask, i], 
                    alpha=0.6, linewidth=0.5, label=f'Env {env}')
    
    axes[i].set_ylabel(f'Z{i+1}', fontsize=12, fontweight='bold')
    axes[i].set_title(f'Latent Variable Z{i+1} (True)', fontsize=12)
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Sample Index', fontsize=12)
plt.tight_layout()
plt.savefig('./outputs/02_true_latents.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Latent causal data generated")

# %% [markdown]
# ## Step 3: Mix Latents into Observations
# 
# Apply a nonlinear mixing function to create observed variables

# %%
def nonlinear_mixing(latents, n_observed=6):
    """Mix latents nonlinearly into observations"""
    
    n_samples, n_latents = latents.shape
    observations = np.zeros((n_samples, n_observed))
    
    # Create random mixing weights
    mixing_matrix = np.random.randn(n_latents, n_observed) * 0.5
    
    # Linear mixing
    observations = latents @ mixing_matrix
    
    # Add nonlinearities
    observations[:, 0] = np.tanh(observations[:, 0])
    observations[:, 1] = observations[:, 1] ** 2 * np.sign(observations[:, 1])
    observations[:, 2] = np.sin(observations[:, 2])
    observations[:, 3:] = observations[:, 3:]  # Keep some linear
    
    # Add noise
    observations += np.random.randn(*observations.shape) * 0.1
    
    return observations

# Mix the latents
observations = nonlinear_mixing(latents_true, n_observed=6)

print(f"Observations shape: {observations.shape}")
print(f"Observations are now mixed and nonlinear combinations of latents")

# Visualize observations
fig, axes = plt.subplots(3, 2, figsize=(14, 8))
axes = axes.flatten()

for i in range(6):
    for env in range(3):
        mask = environments == env
        axes[i].plot(np.where(mask)[0][:200], observations[mask, i][:200], 
                    alpha=0.6, linewidth=0.5, label=f'Env {env}')
    
    axes[i].set_ylabel(f'X{i+1}', fontsize=10)
    axes[i].set_title(f'Observed Variable X{i+1}', fontsize=10)
    if i == 0:
        axes[i].legend(loc='upper right', fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Observed Variables (Mixed and Nonlinear)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./outputs/03_observations.png', dpi=150, bbox_inches='tight')
plt.show()

# Show correlation between observations
corr_obs = np.corrcoef(observations.T)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_obs, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=[f'X{i+1}' for i in range(6)],
            yticklabels=[f'X{i+1}' for i in range(6)],
            vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation Matrix: Observed Variables\n(Everything is entangled!)', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('./outputs/04_observation_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print("Observations created - latents are now hidden")
