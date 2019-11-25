'''

DBSCAN wrapper over sklearn

@pkmandke

'''

import sklearn.cluster import DBSCAN
import joblib

import os
import time
from datetime import timedelta

class DBSCAN_wrapper:

    def __init__(self, keys, eps, minPts, metric='euclidean', n_jobs=1):

        self.eps, self.minPts = eps, minPts
        self.keys = keys

        self.metric, self.n_jobs = metric, n_jobs

        self.algo_class = DBSCAN(eps=self.eps, min_samples=self.minPts, metric=self.metric, n_jobs=self.n_jobs)

    def fit_model(self, data):

        self.model = self.algo_class.fit(data)
    
    def save_model(self, path=''):

        joblib.dump(self.model, path)
    
