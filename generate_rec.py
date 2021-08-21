import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, csv
import scipy
from scipy.sparse import hstack
from preprocessing import *
from sklearn.decomposition import TruncatedSVD

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
# sparse_data_t = scipy.sparse.load_npz('tokens_sparse_OHE_matrix.npz')
# sparse_data_a = scipy.sparse.load_npz('authors_sparse_OHE_matrix.npz')
# sparse_data_m = scipy.sparse.load_npz('main_topic_sparse_OHE_matrix.npz')
# sparse_data_s = scipy.sparse.load_npz('subtopics_sparse_OHE_matrix.npz')
# sparse_data_s = scipy.sparse.load_npz('publishers_sparse_OHE_matrix.npz')

# svd = TruncatedSVD(n_components= 3000, n_iter= 10, random_state=42, algorithm= 'randomized')
# data = svd.fit_transform(sparse_data_p)
# print(svd.explained_variance_ratio_.sum())
# np.save('publishers_svd_3000.npy', data)

# svd_p = np.load('publishers_svd_800.npy')
# svd_m = np.load('main_topics_svd_100.npy')
# svd_s = np.load('subtopics_svd_100.npy')
# svd_a = np.load('authors_svd_4000.npy')
# svd_t = np.load('tokens_svd_3000.npy')
# svd_tapms = np.concatenate((svd_t, svd_a, svd_p, svd_m, svd_s))


# X = np.load('svd_tapms.npy')
X = np.load('tapms_svd_3000.npy')



items, _, eval = preprocessing()
loc = []

# X = data_svd_3500
rec = {}
neigh = NearestNeighbors(n_neighbors=20, radius=1)
neigh.fit(X)

items.set_index('itemID')
# itemID_dict = {for i in range(items.itemID)}

for e in eval.itemID:
    # _r = str(e) + ":"
    row_ = items.query('itemID == ' + str(e)).index[0]
    distance, neighbors_index = neigh.kneighbors([X[row_]], 100, return_distance=True)
    indexes = neighbors_index[0][:6]
    # if np.prod(indexes.shape) == 0:
    #     print(e)
    rec[e] = items.loc[indexes]
    item_list = items.loc[indexes].itemID[1:6].values.tolist()
    str_list = [str(i) for i in item_list]
    _r = '|'.join(str_list)


    rec[e] = items.loc[indexes]

    loc.append(_r)
recommendations = pd.DataFrame({'itemID' : eval.itemID, 'recommendations' : _r}).set_index('itemID')
recommendations.to_csv('recommendations_2.csv')
np.save('rec_2', rec)
print(loc)
