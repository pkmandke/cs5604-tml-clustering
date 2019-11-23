'''
Trigger script
'''


from pre_process import Doc2vec_wrapper, extract_mapped_doc2vecs
import kmeans
import gensim

import time
from datetime import timedelta
import os

SAVE_PATH = '../obj/tobacco/doc2vec/Doc2vec_wrapper_'
TB_PATH = '/mnt/ceph/shared/tobacco/data/1million_raw/'

def main():
    t1 = time.monotonic()

    doc2vec_model = Doc2vec_wrapper(tb_path=TB_PATH, n_docs=len(os.listdir(TB_PATH)))
    
    doc2vec_model.generate_tokens()
    
    doc2vec_model.load_model_and_build_vocab(vector_size=128, dm=1, dm_mean=1, dbow_words=0, epochs=25, workers=10, min_count=2)
    
    doc2vec_model.train()
    
    doc2vec_model.save_model(path=SAVE_PATH)

    print("Time taken {}s".format(timedelta(time.monotonic() - t1)))

main()
