import matplotlib.pyplot as plt

def visualize_clusters(embeddings_2d, cluster_labels, title="Clusters visualisés en 2D"):
    """Affiche les clusters en 2D avec un scatter plot."""
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

