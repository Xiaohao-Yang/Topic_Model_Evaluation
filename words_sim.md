# WALM score functions

WALM score functions compute the similarity between two sets of words.


```python
from walm import walm_overlap, walm_synset, walm_ot, walm_oa
import gensim.downloader as api # we use gensim word embedding models
```

    [nltk_data] Downloading package wordnet to /home/xiaohao/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!



```python
words1 = ['us', 'au']
words2 = ['america', 'australia', 'people']
```

## Overlap-based score functions


```python
# overlap based scores
print('walm overlap: ', walm_overlap(words1, words2))
print('walm synset: ', walm_synset(words1, words2))
```

    walm overlap:  0.0
    walm synset:  0.2


walm_overlap measures the overlap between two sets of words, while walm_synset extends this by considering synset overlap between different words.

## Embedding-based score functions


```python
# load word embedding model
print('Loading glove model ...')
embedding_model = api.load("glove-wiki-gigaword-50")
print('Loading done!')
```

    Loading glove model ...
    Loading done!



```python
# optimal assignment
print('walm optimal assignment: ', walm_oa(words1, words2, embedding_model))
```

    walm optimal assignment:  0.978635346639759


walm_oa solves an optimal assignment problem between word set 1 and word set 2.


```python
# optimal transport
word_dis1 = {'words': words1, 'weights': [0.5, 0.5]}
word_dis2 = {'words': words2, 'weights': [0.1, 0.1, 0.8]}
print('walm optimal transport: ', walm_ot(word_dis1, word_dis2, embedding_model))
```

    walm optimal transport:  0.5193005172391889


walm_ot solves an optimal transport problem between word distribution 1 and word distribution 2.

For further details of the computation, please refer to the implmenetation [here] or the paper [here]
