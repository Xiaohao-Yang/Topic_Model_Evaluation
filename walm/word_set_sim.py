from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import numpy as np
import ot
nltk.download('wordnet')


def walm_overlap(word_list1, word_list2, root='stem', norm='union'):
    if root == 'stem':
        stemmer = PorterStemmer()
    elif root == 'lemma':
        stemmer = WordNetLemmatizer()

    words1 = {stemmer.stem(word) for word in word_list1}
    words2 = {stemmer.stem(word) for word in word_list2}

    sim = len(words1.intersection(words2))

    if norm == 'sum':
        sim = sim / (len(words1) + len(words2))
    elif norm == 'union':
        sim = sim / (len(words1.union(words2)))

    return sim


def walm_synset(word_list1, word_list2, root='stem', norm='union'):
    synset_overlap = 0
    for i in range(len(word_list1)):
        for j in range(len(word_list2)):
            synset1 = wordnet.synsets(word_list1[i])
            synset2 = wordnet.synsets(word_list2[j])
            if len(set(synset1).intersection(set(synset2))) > 0:
                synset_overlap += 1

    if root == 'stem':
        stemmer = PorterStemmer()
    elif root == 'lemma':
        stemmer = WordNetLemmatizer()

    words1 = {stemmer.stem(word) for word in word_list1}
    words2 = {stemmer.stem(word) for word in word_list2}

    if norm == 'sum':
        sim = synset_overlap / (len(words1) + len(words2))
    elif norm == 'union':
        sim = synset_overlap / (len(words1.union(words2)))

    return sim


def construct_cost_matrix(words1, words2, embedding_model):
    cost = np.zeros(shape=(len(words1), len(words2)))
    for i in range(len(words1)):
        for j in range(len(words2)):
            try:
                word_embedding1 = embedding_model[words1[i].strip()]
            except:
                print('Can not find word in the vocabulary of the embedding model: ', words1[i])
                quit()
            try:
                word_embedding2 = embedding_model[words2[j].strip()]
            except:
                print('Can not find word in the vocabulary of the embedding model: ', words2[j])
                quit()

            dist = cosine(word_embedding1, word_embedding2)
            cost[i, j] = dist

    return cost


def walm_oa(word_list1, word_list2, embedding_model):
    word_list1 = [w.lower() for w in word_list1]
    word_list2 = [w.lower() for w in word_list2]

    try:
        embedding_model['cat']
    except:
        print('Cat not get word embedding from the embedding model')
        print('We support gensim models only')
        quit()

    cost = construct_cost_matrix(word_list1, word_list2, embedding_model)
    row_ind, col_ind = linear_sum_assignment(cost)
    sim = cost[row_ind, col_ind].sum()

    return sim


def walm_ot(word_dis1, word_dis2, embedding_model):
    word_list1, word_list2 = word_dis1['words'], word_dis2['words']
    word_weights1, word_weights2 = word_dis1['weights'], word_dis2['weights']

    word_list1 = [w.lower() for w in word_list1]
    word_list2 = [w.lower() for w in word_list2]

    try:
        embedding_model['cat']
    except:
        print('Cat not get word embedding from the embedding model')
        print('We support gensim models only')
        quit()

    cost = construct_cost_matrix(word_list1, word_list2, embedding_model)
    sim = ot.emd2(word_weights1, word_weights2, cost)

    return sim
