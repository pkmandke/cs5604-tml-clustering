'''

The BIRCH clustering technique

@pkmandke
'''

from sklearn.cluster import Birch
import joblib

class BIRCH:

    def __init__(self, doc_list, threshold=1.0, branching_factor=50, n_clusters=500, compute_labels=True, copy=False):

        self.doc_list = doc_list
        self.threshold, self.branching_factor, self.n_clusters = threshold, branching_factor, n_clusters

        self.compute_labels, self.copy = compute_labels, copy

        self.model = Birch(threshold=self.threshold, branching_factor=branching_factor, n_clusters=n_clusters, compute_labels=compute_labels, copy=copy)

    def fit(self, data):

        self.fit_model = self.model.fit(data)
    
    def save(self, path=''):

        joblib.dump(self, path)