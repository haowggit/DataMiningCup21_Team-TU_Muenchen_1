import sklearn.preprocessing
import pandas as pd
import numpy as np
import scipy
import umap
from sklearn.decomposition import TruncatedSVD


def learn_manifold(x_data, umap_min_dist = 0.00, umap_metric = 'euclidean', umap_dim = 10, umap_neighbors = 30):
    md = float(umap_min_dist)
    umap_learning = umap.UMAP(random_state=0, metric=umap_metric, n_components=umap_dim, n_neighbors=umap_neighbors,
              min_dist=md)
    umap_learning.fit_transform(x_data)
    return umap




# load data
sparse_data = scipy.sparse.load_npz('OHE_sparse_matrix.npz')

svd = TruncatedSVD(n_components= 500, n_iter= 10, random_state=42, algorithm= 'randomized')
svd.fit(sparse_data)

print(svd.explained_variance_ratio_.sum())
new_data = svd.fit_transform(sparse_data)

reduced_data = learn_manifold(new_data, umap_dim=250)