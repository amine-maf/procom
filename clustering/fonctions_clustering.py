def agglomerative_clustering(reduced_embeddings, n_clusters):
    """
    Applique l'Agglomerative Clustering avec un nombre fixe de clusters.
    
    :param reduced_embeddings: embeddings
    :param n_clusters: Nombre de clusters
    :return: Labels prédits
    """
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg_clustering.fit_predict(reduced_embeddings)
    return agg_labels

def gaussian_mixture(reduced_embeddings, n_clusters):
    """
    Applique le modèle de mélange gaussien (GMM) avec un nombre fixe de clusters.
    
    :param reduced_embeddings: embeddings
    :param n_clusters: Nombre de clusters
    :return: Labels prédits
    """
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(reduced_embeddings)
    return gmm_labels