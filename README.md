# CS5604 Information Storage and Retrieval - Fall 2019

**Team**: Text Analytics and Machine Learning

**Instructor**: [Dr. Edward Fox](https://fox.cs.vt.edu/)

**Update**: A beta version of the front-end website for querying both the ETD and Tobacco corpora is accessible [here](http://2001.0468.0c80.6102.0001.7015.a60f.cf44.ip6.name:3000/)! It is running based on Elastic Search via Ceph on the Kubernetes cluster at [cs.vt.edu](cs.vt.edu).

This repository contains code for clustering 2 large text corpora for efficient information retrieval. Please see the [data](https://github.com/pkmandke/cs5604-tml-clustering/tree/master/data) directory for details regarding the ETD and the Tobacco corpora.

## Bare Beginnings

To begin with we pre-process the ETD and Tobacco text corpora. Refer to [this](https://github.com/pkmandke/cs5604-tml-clustering/blob/f80aa07df09409f517ea4b81ad3e8b3f982f7257/src/pre_process.py#L48) method in the pre-process script for implementation details. The pre-processing phase involves the following steps:

1. Convert the entire document to lower case.
2. Remove punctuations if any.
3. Tokenize the document using a combination of an improved Tree Bank Tokenizer along with a Punkt Sentence Tokenizer. These are packages together within the [word_tokenizer](https://www.nltk.org/api/nltk.tokenize.html) utility by NLTK.
4. Remove all stopwords that appear in the default stopwords list provided by nltk in Section 4.1 [here](https://www.nltk.org/book/ch02.html).
5. Apply nltk's default [PorterStemmer](https://www.nltk.org/_modules/nltk/stem/porter.html) for stemming the words.

## From Documents to Vectors

We use [Doc2vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf), a neural network based vector embedding computation technique proposed by Quoc Le and Tomas Mikolov. In particular, we use the distributed memory setting to train 1 vector per document for the ETDs as well as the TSRs. We only train the document vectors and do not backprop into the word vector embeddings. The size of the document embeddding has been chosen to be 128 based on heuristics. In so far as implementation is concerned, we use Gensim's [Doc2vec class](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec) for training the models. Note that in the case of ETDs, we train the document vectors purely on the basis of the abstracts of the documents which are contained in the 'abstract-description' field of the metadata. For more details, refer to the [ETD results](src/ETD_results.ipynb). Refer to [this](https://github.com/pkmandke/cs5604-tml-clustering/blob/f80aa07df09409f517ea4b81ad3e8b3f982f7257/src/pre_process.py#L22) script for details.

## K for Clustering

We implement and evaluate the following clustering techniques for the ETD and/or Tobacco corpora. 

1. K-Means clustering
2. Agglomerative Clustering
3. DBSCAN
4. BIRCH


> Incomplete. #TODO: Results and implementation details.

# Authors

[Prathamesh Mandke](https://pkmandke.github.io/) - I am the maintainer of this repository. 

[Sharvari Chougule]()

[Adheesh Juvekar]()

# Acknowledgements

To begin with, we would like to thank Dr. Fox for his constant support and encouragement towards taking this work towards completion. We are grateful to all the teams from the CS-5604 class for synergistically ensuring the success of getting the information retrieval system up and running (A beta version is accessible [here](http://2001.0468.0c80.6102.0001.7015.a60f.cf44.ip6.name:3000/)). Thanks are also due to [Dr. David Townsend](https://management.pamplin.vt.edu/faculty/directory/townsend-david.html), Assistant Professor at the Pamplin School of Business at Virginia Tech for his insights into understanding the Tobacco corpus to better direct the clustering efforts. 