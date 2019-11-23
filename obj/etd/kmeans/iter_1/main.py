'''
Trigger script
'''


from pre_process import Doc2vec_wrapper, extract_mapped_doc2vecs
import kmeans
import gensim

import time
from datetime import timedelta

def main():
    t1 = time.monotonic()

    model = gensim.models.doc2vec.Doc2Vec.load('../obj/doc2vec/abstracts_etd_doc2vec_5000_docs')
    
    doc_vectors, keys = extract_mapped_doc2vecs(model)
    
    km_obj = kmeans.Kmeans(doc_list=keys, n_clusters=10, init='k-means++', n_init=3, n_jobs=5, random_state=42, verbose=1, algorithm='full')

    km_obj.fit(doc_vectors)

    km_obj.save('abstracts_etd_doc2vec_5000_docs_kmeans.sav')


    print("Time taken {}s".format(timedelta(time.monotonic() - t1)))

main()
