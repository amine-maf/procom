# imports

from clustering.agg_clustering import *
from clustering.mesures_clustering import *
from results.affichage_messages_par_cluster import *
from utils.reduction_dimesion import *


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def main():
    # lecture du fichier csv
    df = load .....
    # reduction des dimensions
    reduced_embeddings = reduce_embeddings(df, n_components=50, random_state=42)
    # clustering
    labels = agglomerative_clustering(reduced_embeddings, n_clusters=14)
    # metriques de clustering
    silhouette_score = compute_silhouette_scores(reduced_embeddings, labels)
    # afficher les resultats
    messages_par_cluster= afficher_messages_par_cluster(df,labels)


if __name__ == "__main__":
    main()
