'''

Agglomerative clustering

'''
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering as AC

from datetime import timedelta
import time

import joblib

class Agglo_clus:
    
    def __init__(self, feature_matrix, doc_names, num_clus=20, linkage='ward', affinity='euclidean', iter_='1'):
        
        self.feature_matrix = feature_matrix
        self.iter = iter_
        self.doc_names = doc_names;
        self.num_clus = num_clus
        self.linkage = linkage
        self.affinity = affinity
        
    def clusterize(self):
        
        print("Starting clustering...");
        t1 = time.monotonic()
        agg_clus = AC(n_clusters=self.num_clus, affinity=self.affinity, linkage=self.linkage)
        self.predictions = agg_clus.fit_predict(self.feature_matrix)
        print("Done training in {}s".format(timedelta(seconds=time.monotonic() - t1)))
        
    def save(self, name='agglo_clus_1.sav'):
        joblib.dump(self, '../obj/agglo_clus/iter_' + self.iter + '/' + name)