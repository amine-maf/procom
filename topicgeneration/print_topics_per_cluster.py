def print_topics_per_cluster(df, labels):
    # Ajouter labels comme colonne 'cluster' dans df
    df['cluster'] = labels

    print("\nTopics par cluster pour l'ensemble des données :")
    for cluster in sorted(df['cluster'].unique()):
        print(f"\nCluster {cluster}:")
        messages_cluster = df[df['cluster'] == cluster]['message'].tolist()
        topics = generate_topics_for_cluster(messages_cluster)
        print("Topics générés:", topics)