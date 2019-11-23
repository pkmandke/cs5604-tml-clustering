'''

Pre-process ETD docs to retrieve abstracts from ETD JSONs and Tobacco SRs

Author: Prathamesh Mandke

Date: 11/21/2019

'''

# Imports

import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os
import string

class Doc2vec_wrapper:

    def __init__(self, json_path=None, tb_path=None, n_docs=-1):

        self.tb_path = tb_path        
        self.json_path = json_path
        
        if json_path:
            self.dframe = pd.read_json(json_path, orient=str, lines=True)[['identifier-uri', 'description-abstract']]
            if n_docs == -1:
                self.tot_len = len(self.dframe)
            else:
                self.tot_len = n_docs
        
        if tb_path:
            if n_docs == -1:
                self.tot_len = os.listdir(self.tb_path)
            else:
                self.tot_len = n_docs

        print("Initializing preprocessor object. Downloading NLTK stopwords...")

        nltk.download('punkt')
        nltk.download('stopwords')
     

    def __tokenize_string(self, document):

        if type(document) == str:
                tokens = nltk.word_tokenize(document.lower().translate(str.maketrans('', '', string.punctuation)))
        else:
            return None
            
        stems = []
        
        for item in tokens:
            if item in stopwords.words():
                continue
            stems.append(PorterStemmer().stem(item))

        #tokens = gensim.utils.simple_preprocess(document)
        
        return stems

    def __abstracts_data_generator(self, document=None):
        ''' Deprecated. Can be used for ETDs'''
        for idx in range(self.tot_len):
            
            document = self.dframe.iloc[idx]['description-abstract']

            if type(document) == str:
                tokens = nltk.word_tokenize(document.lower().translate(str.maketrans('', '', string.punctuation)))
            else:
                continue
            
            stems = []
            
            for item in tokens:
                if item in stopwords.words():
                    continue
                stems.append(PorterStemmer().stem(item))

            #tokens = gensim.utils.simple_preprocess(document)
            
            yield gensim.models.doc2vec.TaggedDocument(stems, [self.dframe.iloc[idx]['identifier-uri']])
            
            
    def __tobacco_data_gen(self):
        
        for doc in os.listdir(self.tb_path):

            document = open(os.path.join(self.tb_path, doc), 'r').read()

            stems = self.__tokenize_string(document)

            if stems:
                yield gensim.models.doc2vec.TaggedDocument(stems, [str(doc)])
            else:
                continue
    
    def generate_tokens(self):

        if self.json_path:
            self.train_corpus = list(self.__abstracts_data_generator())
            del self.dframe
        elif self.tb_path:
            self.train_corpus = list(self.__tobacco_data_gen())
        
    def load_model_and_build_vocab(self, dm=1, dm_mean=1, vector_size=128, min_count=2, epochs=20, workers=1, dbow_words=0):

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, dm=dm, dm_mean=dm_mean, min_count=min_count, dbow_words=dbow_words, epochs=epochs, workers=workers)
        self.model.build_vocab(self.train_corpus)

    def train(self):

        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def save_model(self, path='../obj/abstracts_etd_doc2vec_'):
        path += str(self.tot_len) + '_docs'
        self.model.save(path)
        
def extract_mapped_doc2vecs(model):
    inferred_vector = model.infer_vector(['a', 'b', 'c', 'd'])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    keys_ = [tup[0] for tup in sims]

    doc_vectors = [model.docvecs[key] for key in keys_]

    return doc_vectors, keys_