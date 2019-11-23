'''

Title: K-Means clustering of documents

CS5604@VT

Team: TML

Author: Prathamesh Mandke

Date created: 09/28/2019

'''

# conda activate preproc - deprecated

# conda activate kmeans

# PATHS: TO BE SET Properly

DATA_PATH = '/mnt/ceph/shared/tobacco/text_files_depositions_clean/'
SAVE_PATH = '/mnt/ceph/tml/clustering/kmeans/obj/tobacco/chi_1/'


# Imports

import numpy as np
import nltk
import string
import os
import time
from datetime import timedelta

from sklearn.metrics import calinski_harabasz_score as cb_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
import joblib

doc_cnt = 0

def p_tokenize(text):
    '''Simple Porter stemmer. Code credits: https://www.bogotobogo.com/python/NLTK/tf_idf_with_scikit-learn_NLTK.php'''
    #global doc_cnt
    #doc_cnt += 1
    tokens = nltk.word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))
    stems = []
    #print("Tokenizing Document #{}".format(doc_cnt), end='\r')
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    
    return stems


class TFIDF:

    def __init__(self, data_path=DATA_PATH, tokenizer=p_tokenize, i_input='filename', \
                 decode_error='ignore', strip_accents='ascii', analyzer='word', stop_words='english', \
                min_df=0.0, max_df=0.7, dtype=np.float64):

        self.data_path = data_path
        self.doc_list = [self.data_path + _ for _ in os.listdir(self.data_path)]
        self.input = i_input
        self.decode_error=decode_error
        self.strip_accents = strip_accents
        self.analyzer = analyzer
        self.tokenizer = tokenizer
        
        nltk.download('stopwords', quiet=True, raise_on_error=True)
        stop_words = nltk.corpus.stopwords.words(stop_words)
        self.stop_words = self.tokenizer(' '.join(stop_words))
        print("Stop words: \n {}".format(self.stop_words))
        self.min_df = min_df
        self.max_df = max_df
        self.dtype = dtype
        
        self.doc_iter = Doc_iter(self.doc_list)
        self.vectorizer = TfidfVectorizer(input=self.input, decode_error=self.decode_error, strip_accents=self.strip_accents, \
                                         tokenizer=self.tokenizer, analyzer=self.analyzer, stop_words=self.stop_words, \
                                         min_df=self.min_df, max_df=self.max_df, dtype=self.dtype)

    def fit_transform(self):

        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_iter)

    def save(self, filename):
        joblib.dump(self, filename)


class Kmeans:

    def __init__(self, doc_list, n_clusters=10, init='k-means++', n_init=3, n_jobs=5, random_state=42, verbose=1, algorithm='full', iter_='1'):

        self.doc_list = doc_list
        self.n_clusters = n_clusters
        self.rand_state = random_state
        self.init = init
        self.n_init = n_init
        self.algorithm = algorithm
        self.iter = iter_

        self.km = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, \
                         verbose=verbose, random_state=self.rand_state, algorithm=self.algorithm)

    def fit(self, data):
        '''Fit the kmeans object. Clusters and assignments are stored in \
        attributes of self.km'''

        self.km.fit(data);
        
    def compute_chi_index(self, data):
        print("Computing CH score")
        return cb_score(data.todense(), self.km.labels_)

    def save(self, filename):
        joblib.dump(self, '../obj/kmeans/iter_' + self.iter + '/' + filename)


class Doc_iter:
    '''Document iterator'''
    def __init__(self, doc_list):

        self.doc_list = doc_list
        self.ptr = 0
        self.len_doc_list = len(self.doc_list)

    def __iter__(self):
        return self

    def __next__(self):

        if self.ptr >= self.len_doc_list:
            raise StopIteration

        rvalue = self.doc_list[self.ptr]
        self.ptr += 1

        return rvalue



if __name__ == '__main__':

    '''print("Computing TFIDF vectors:")
    t1 = time.monotonic()
    tfidf = TFIDF()
    tfidf.fit_transform()
    print("Done in {}s.".format(timedelta(seconds=time.monotonic() - t1)))
    tfidf.save(SAVE_PATH + 'TFIDF_1.sav')
   '''

    tfidf = joblib.load(SAVE_PATH + 'TFIDF_1.sav')
    km = Kmeans(doc_list=tfidf.doc_list.copy())
    data = tfidf.tfidf_matrix
    del tfidf
    print("Computing Kmeans clusters:")
    t1 = time.monotonic()
    km.fit(data)
    print("Done in {}s".format(timedelta(seconds=time.monotonic() - t1)))
    km.save(SAVE_PATH + 'Kmeans_1.sav')
