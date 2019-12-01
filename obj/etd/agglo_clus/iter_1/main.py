'''
Trigger script
'''


from pre_process import Doc2vec_wrapper, extract_mapped_doc2vecs
import kmeans
import gensim
import dbscan
import utils
from agglo_clus import Agglo_clus

import time
from datetime import timedelta
import os


ITER = '1'
SAVE_PATH = '../obj/etd/agglo_clus/iter_' + ITER + '/agglo_clus_obj.sav'
TB_PATH = '/mnt/ceph/shared/tobacco/data/1million_raw/'
DOCVEC_PATH = '../obj/etd/doc2vec/abstracts_etd_doc2vec_all_docs30961_docs'

def main():
    t1 = time.monotonic()

    docvec_model = gensim.models.doc2vec.Doc2Vec.load(DOCVEC_PATH)

    doc_vectors, keys = extract_mapped_doc2vecs(docvec_model)

    model = Agglo_clus(doc_vectors, keys, num_clus=500, linkage='ward', affinity='euclidean', iter_=ITER)

    model.clusterize()

    model.save(name=SAVE_PATH)
    
    print("Time taken {}s".format(timedelta(seconds=time.monotonic() - t1)))

main()
