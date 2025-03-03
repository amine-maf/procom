def agglomerative_clustering(reduced_embeddings, cluster_range=range(2, 21)):
    """
    Teste l'Agglomerative Clustering avec différents nombres de clusters.
    
    :param reduced_embeddings: Matrice des embeddings réduits
    :param cluster_range: Intervalle du nombre de clusters à tester
    :return: Liste des labels prédits
    """
    clustering_results = {}
    for n_clusters in cluster_range:
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = agg_clustering.fit_predict(reduced_embeddings)
        clustering_results[n_clusters] = agg_labels
    return clustering_results

def gaussian_mixture(reduced_embeddings, cluster_range=range(2, 21)):
    """
    Teste le modèle de mélange gaussien (GMM) avec différents nombres de clusters.
    
    :param reduced_embeddings: Matrice des embeddings réduits
    :param cluster_range: Intervalle du nombre de clusters à tester
    :return: Liste des labels prédits
    """
    clustering_results = {}
    for n_clusters in cluster_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(reduced_embeddings)
        clustering_results[n_clusters] = gmm_labels
    return clustering_results
