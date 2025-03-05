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