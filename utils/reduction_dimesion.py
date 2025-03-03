import numpy as np
import umap.umap_ as umap
import ast

def reduce_embeddings(df, n_components=50, random_state=42):
    """
    Réduit la dimension des embeddings en utilisant UMAP.

    :param df: DataFrame contenant une colonne 'embedding_new' avec les embeddings
    :param n_components: Nombre de dimensions pour la réduction (par défaut 50)
    :param random_state: Seed pour la reproductibilité (par défaut 42)
    :return: Matrice des embeddings réduits (reduced_embeddings)
    """
    # Convertir les embeddings en format numpy si nécessaire
    df['embedding_new'] = df['embedding_new'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Créer une matrice numpy à partir des embeddings
    embeddings_matrix = np.stack(df['embedding_new'].values)
    
    # Appliquer UMAP pour réduire la dimension
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced_embeddings = umap_model.fit_transform(embeddings_matrix)
    
    return reduced_embeddings