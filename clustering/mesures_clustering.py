def compute_silhouette_scores(reduced_embeddings, clustering_results):
    """
    Calcule les scores de silhouette pour différents nombres de clusters.
    
    :param reduced_embeddings: Matrice des embeddings réduits
    :param clustering_results: Dictionnaire contenant les labels prédits pour chaque nombre de clusters
    :return: Dictionnaire des scores de silhouette
    """
    silhouette_scores = {}
    for n_clusters, labels in clustering_results.items():
        silhouette_avg = silhouette_score(reduced_embeddings, labels)
        silhouette_scores[n_clusters] = silhouette_avg
        print(f"Nombre de clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.2f}")
    return silhouette_scores