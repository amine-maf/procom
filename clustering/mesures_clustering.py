from sklearn.metrics import silhouette_score

def compute_silhouette_scores(reduced_embeddings, labels):
    """
    Calcule les scores de silhouette pour différents nombres de clusters.
    
    :param reduced_embeddings: Matrice des embeddings réduits
    :param labels: les labels prédits par les algos de clustering
    :return: silhouette score
    """
    silhouette_avg = silhouette_score(reduced_embeddings, labels)
    return silhouette_avg


def compute_dunn_index(reduced_embeddings, labels):
    """
    Calcule l'index de Dunn pour évaluer la qualité du clustering.
    
    :param reduced_embeddings: Matrice des embeddings réduits
    :param labels: Labels prédits par les algorithmes de clustering
    :return: Index de Dunn
    """
    unique_labels = np.unique(labels)
    # calcul des distances intra-cluster
    intra_cluster_distances = []
    for label in unique_labels:
        cluster_points = reduced_embeddings[labels == label]
        if len(cluster_points) > 1:
            distances = pdist(cluster_points)
            intra_cluster_distances.append(np.max(distances))
    #calcul des distances inter-cluster 
    inter_cluster_distances = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                points_i = reduced_embeddings[labels == label_i]
                points_j = reduced_embeddings[labels == label_j]
                distances = pdist(np.vstack([points_i, points_j]))
                inter_cluster_distances.append(np.min(distances))   
    #calcul de l'index de Dunn
    dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
    return dunn_index