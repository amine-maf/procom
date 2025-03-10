from sklearn.decomposition import PCA



def reduce_embeddings_2d(embeddings, n_components=2, random_state=42):
    """Réduit les embeddings à 2 dimensions avec PCA."""
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(embeddings)