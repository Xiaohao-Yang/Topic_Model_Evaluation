from nltk.corpus import wordnet
from scipy.spatial.distance import cosine
import numpy as np
from scipy.optimize import linear_sum_assignment
from nltk.stem import PorterStemmer
import nltk
import ot
nltk.download('wordnet')


def make_uni_dis(top_words_list):
    dis_list = []
    for i in range(len(top_words_list)):
        N = len(top_words_list[i])
        dis = np.array([1/N]*N, dtype=np.float64)
        dis_list.append(dis)

    return dis_list


def construct_cost_matrix(words1, words2, embedding_model):
    cost = np.zeros(shape=(len(words1), len(words2)))
    for i in range(len(words1)):
        for j in range(len(words2)):
            try:
                word_embedding1 = embedding_model[words1[i].strip()]
            except:
                print('something wrong with words from topic model')
                print(words1[i])
                quit()
            try:
                word_embedding2 = embedding_model[words2[j].strip()]
            except:
                print('something wrong with words from LLM')
                print(words2[j])
                quit()

            dist = cosine(word_embedding1, word_embedding2)
            cost[i, j] = dist

    return cost


def synset_similarity(words1, words2):
    count = 0
    for i in range(len(words1)):
        for j in range(len(words2)):
            synset1 = wordnet.synsets(words1[i])
            synset2 = wordnet.synsets(words2[j])
            if len(set(synset1).intersection(set(synset2))) > 0:
                count += 1

    return count


def oa_similarity(cost):
    row_ind, col_ind = linear_sum_assignment(cost)
    sim = cost[row_ind, col_ind].sum()

    return sim


def ot_similarity(distribution1, distribution2, cost):
    sim = ot.emd2(distribution1, distribution2, cost)

    return sim


# word_list1, word_list2: a list of words, for each document
def wordlists_similarity_overlap(word_list1, word_list2, method='synset'):
    total = 0
    N = len(word_list1)
    stemmer = PorterStemmer()
    # lemmatizer = WordNetLemmatizer()

    for i in range(N):
        if method == 'overlap':
            words1 = {stemmer.stem(word) for word in word_list1[i]}
            words2 = {stemmer.stem(word) for word in word_list2[i]}
            sim = len(words1.intersection(words2))
            sim = sim / (len(words1) + len(words2))

        elif method == 'synset':
            sim = synset_similarity(word_list1[i], word_list2[i])
            sim = sim / (len(word_list1[i]) + len(word_list2[i]))

        total += sim

    return total/N


# word_list1, word_list2: a list of words, for each document
# distribution1, distribution2: a list of words' probability mass, for each document
# embedding_model: the model to obtain word embeddings
def wordlists_similarity_distance(word_list1, distributions1, word_list2, distributions2, embedding_model, method='OA'):
    total = 0
    N = len(word_list1)

    for i in range(N):
        cost = construct_cost_matrix(word_list1[i], word_list2[i], embedding_model)

        if method == 'OA':
            sim = oa_similarity(cost)

        elif method == 'OT':
            sim = ot_similarity(distributions1[i], distributions2[i], cost)

        total += sim

    return total/N