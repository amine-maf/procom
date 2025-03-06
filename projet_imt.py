# imports

from clustering.agg_clustering import *
from clustering.mesures_clustering import *
from results.affichage_messages_par_cluster import *
from utils.reduction_dimesion import *


def main():
    # lecture du fichier csv
    df = pd.read_csv("/Users/mac/Desktop/topic_modeling/df_user_messages_new.csv")
    # reduction des dimensions
    reduced_embeddings = reduce_embeddings(df, n_components=50, random_state=42)
    # clustering
    labels = agglomerative_clustering(reduced_embeddings, n_clusters=14)
    # metriques de clustering
    silhouette_score = compute_silhouette_scores(reduced_embeddings, labels)
    # afficher les resultats
    messages_par_cluster= afficher_messages_par_cluster(df,labels)

    return messages_par_cluster


if __name__ == "__main__":
    messages_par_cluster = main()
    print(messages_par_cluster)