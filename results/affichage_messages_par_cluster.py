def afficher_messages_par_cluster(df,labels):
    """
    Ajoute les labels de cluster à la table et retourne les messages par cluster.

    :param labels: Liste des labels de cluster déjà générés
    :param df: DataFrame contenant la colonne 'message'
    :return: Dictionnaire {cluster: [messages]}
    """
    df['cluster'] = labels
    messages_par_cluster = {}
    
    # Remplir le dictionnaire avec les messages de chaque cluster
    for cluster in sorted(df['cluster'].unique()):
        messages = df[df['cluster'] == cluster]['message'].tolist()
        messages_par_cluster[cluster] = messages
    
    return messages_par_cluster