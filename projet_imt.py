# imports

from clustering.agg_clustering import *
from clustering.mesures_clustering import *
from results.print_messages_per_cluster import *
from utils.reduction_dimesion import *
from utils.reduction_dimension_2d import *
from results.visualize_clusters import *
from topicgeneration.generate_topics import *
from topicgeneration.print_topics_per_cluster import *



#clé API
openai.api_key = "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"



def main():
    # lecture du fichier csv
    df = pd.read_csv("/Users/mac/Desktop/topic_modeling/df_user_messages_new.csv")
    # reduction des dimensions
    reduced_embeddings = reduce_embeddings(df, n_components=100, random_state=42)
    # rediction des dimension en 2d pour la visualisation
    reduced_embeddings_2d = reduce_embeddings_2d(reduced_embeddings, n_components=2, random_state=42)
    # clustering
    cluster_labels = agglomerative_clustering(reduced_embeddings, n_clusters=11)
    # visualisation des clusters en 2D
    visualize_clusters(reduced_embeddings_2d, cluster_labels, title="Clusters visualisés en 2D")
    # metriques de clustering
    silhouette_score = compute_silhouette_scores(reduced_embeddings, cluster_labels)
    dunn_index = compute_dunn_index(reduced_embeddings, cluster_labels)
    # afficher les resultats
    messages_par_cluster= print_messages_per_cluster(df, cluster_labels)
    # afficher les messages avec les topics
    resultats= print_topics_per_cluster(df, cluster_labels):


    return resultats


if __name__ == "__main__":
    resultats = main()
    print(resultats)