'''
Trigger script
'''


from pre_process import Doc2vec_wrapper, extract_mapped_doc2vecs
import kmeans
import gensim
import dbscan
import utils

import time
from datetime import timedelta
import os


ITER = '1'
SAVE_PATH = '../obj/etd/DBSCAN/iter_' + ITER + '/dbscan_wrapper.sav'
TB_PATH = '/mnt/ceph/shared/tobacco/data/1million_raw/'
DOCVEC_PATH = '../obj/etd/doc2vec/abstracts_etd_doc2vec_all_docs30961_docs'

def main():
    t1 = time.monotonic()

    docvec_model = gensim.models.doc2vec.Doc2Vec.load(DOCVEC_PATH)

    doc_vectors, keys = extract_mapped_doc2vecs(docvec_model)

    model = dbscan.DBSCAN_wrapper(keys=keys, eps=7, minPts=4, metric='euclidean', n_jobs=10)

    model.fit_model(doc_vectors)

    model.save_model(path=SAVE_PATH)
    
    print("Time taken {}s".format(timedelta(time.monotonic() - t1)))

main()
