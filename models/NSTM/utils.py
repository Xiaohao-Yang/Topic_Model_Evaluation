import numpy as np
from operator import itemgetter
import torch
import math
from scipy.special import softmax as softmax_np


def get_topic_words(beta, voc, save_file, topn=10):
    K = beta.shape[0]
    idx2token = {index: word for index, word in enumerate(voc)}
    topic_list = []
    for i in range(K):
        top_word_idx = np.argsort(beta[i, :])[::-1]
        top_word_idx = top_word_idx[0:topn]
        top_words = itemgetter(*top_word_idx)(idx2token)
        topic_list.append(top_words)

    if save_file:
        with open(save_file, 'w') as file:
            for item in topic_list:
                file.write(' '.join(item) + '\n')
            file.close()

    return topic_list


def get_theta(model, dataset, batch_size=1000):
    model.eval()

    N = dataset.shape[0]
    dataset = torch.from_numpy(dataset).cuda()
    nb_batches = int(math.ceil(float(N) / batch_size))

    theta = np.zeros((N, model.K))
    for batch in range(nb_batches):
        start, end = batch_indices(batch, N, batch_size)
        X = dataset[start:end]
        X_theta = model(X)
        theta[start:end] = X_theta.detach().cpu().numpy()

    return theta


def get_doc_word(beta, theta, voc, topn=10):
    doc_word = softmax_np(np.matmul(theta, beta), -1)
    voc = {index: word for index, word in enumerate(voc)}

    # sort and keep only topn
    indices = np.argsort(-doc_word, axis=1)
    sorted_arr = np.take_along_axis(doc_word, indices, axis=1)
    top_n_sorted_arr = sorted_arr[:, :topn]
    top_n_indices = indices[:, :topn]

    top_words_list = []
    for i in range(top_n_indices.shape[0]):
        doc_top_words = itemgetter(*top_n_indices[i])(voc)
        top_words_list.append(doc_top_words)

    top_n_sorted_arr = top_n_sorted_arr / top_n_sorted_arr.sum(1)[:, np.newaxis]

    return top_words_list, top_n_sorted_arr


def perplexity(model, test_1, test_2, beta):
    model.eval()

    theta_test1 = get_theta(model, test_1)
    sums_2 = test_2.sum(1)
    doc_word = softmax_np(np.matmul(theta_test1, beta), -1)
    preds = np.log(doc_word)
    recon_loss = -(preds * test_2.data).sum(1)
    loss = recon_loss / (sums_2 + 1e-10)
    loss = np.nanmean(loss)
    ppl = round(math.exp(loss), 1)

    return ppl


def get_voc_embeddings(voc, embedding_model):
    word_embeddings = []
    for v in voc:
        word_embeddings.append(embedding_model[v])
    word_embeddings = np.array(word_embeddings)
    return word_embeddings


def batch_indices(batch_nb, data_length, batch_size):
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end