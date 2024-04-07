import re
import numpy as np
from gensim.matutils import Sparse2Corpus
import math
from scipy import sparse
from operator import itemgetter


def get_phi(lda_model):
    phi = lda_model.get_topics()
    return phi


def format_doc_topic(doc_topic_lda, k):
    N = len(doc_topic_lda)
    K = k
    doc_topic_np = np.zeros(shape=(N,K))

    for i in range(N):
        doc = doc_topic_lda[i]

        if len(doc) != K:
            idx_k = [tup[0] for tup in doc]
            for m in range(K):
                if m not in idx_k:
                    new_tup = (m, 0.0)
                    doc.append(new_tup)

        doc_sorted = sorted(doc, key=lambda tup: tup[0])

        for j in range(K):
            doc_topic_np[i, j] = doc_sorted[j][1]

    return doc_topic_np


def get_theta(model, data):
    data = sparse.csr_matrix(data)
    corpus = Sparse2Corpus(data.transpose())

    doc_topic = model[corpus]
    theta = format_doc_topic(doc_topic, model.num_topics)

    return theta


def get_topic_words(n_topic, lda_model, voc, save_file=None, topn=10):
    topics = lda_model.show_topics(n_topic, num_words=topn)
    topic_list = []

    for topic in topics:
        re_list = re.findall(r'(?<=").*?(?=")', str(topic[1]))
        word_list = []
        for item in re_list:
            if '*' not in item:
                word_list.append(item)
        topic_list.append(word_list)

    if save_file:
        with open(save_file, 'w') as file:
            for item in topic_list:
                file.write(' '.join(item) + '\n')
            file.close()

    return topic_list


def get_doc_word(phi, theta, voc, topn=10):
    # doc over word distribution
    doc_word = np.dot(theta, phi)

    # get top words from doc-word distribution
    doc_top_words = []
    for i in range(doc_word.shape[0]):
        idx = np.argsort(doc_word[i])[::-1][0:topn]
        doc_top_words.append(list(itemgetter(*idx)(voc)))

    # get top probabilities from doc-word distribution
    sorted_indices = np.argsort(-doc_word, axis=1)
    top_doc_word_dis = doc_word[np.arange(doc_word.shape[0])[:, None], sorted_indices[:, :topn]].astype(
        np.float64)
    # normalise as distribution
    doc_top_dis = top_doc_word_dis / top_doc_word_dis.sum(1)[:, np.newaxis]

    return doc_top_words, doc_top_dis


def perplexity(trained_lda, test_1, test_2, batch_size=1000):
    beta = get_phi(trained_lda)
    beta += np.finfo(np.float64).eps

    n_test = test_1.shape[0]
    num_batches = int(math.ceil(n_test / batch_size))

    acc_loss = 0    # batch ppl in total
    count = 0       # number of batches

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        test1_batch = test_1[start:end, :]
        test2_batch = test_2[start:end, :]

        # test1 to get theta
        test1_batch_corpus = Sparse2Corpus(test1_batch.transpose())
        theta = trained_lda[test1_batch_corpus]
        theta = format_doc_topic(theta, beta.shape[0])

        # get predition loss using test2
        if sparse.isspmatrix(test2_batch):
            test2_batch = test2_batch.toarray()
        test2_batch = test2_batch.astype('float64')

        sums_2 = test2_batch.sum(1)
        preds = np.log(np.matmul(theta, beta))
        recon_loss = -np.nansum((preds * test2_batch), 1)
        loss = recon_loss / sums_2
        loss = np.nanmean(loss)

        acc_loss += loss
        count += 1

    cur_loss = acc_loss / count
    ppl_dc = round(math.exp(cur_loss), 1)

    return ppl_dc