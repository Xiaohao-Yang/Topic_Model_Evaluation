from operator import itemgetter
import numpy as np
from scipy.special import softmax as softmax_np
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import softmax
import math


def get_phi(model):
    phi = model.dec_vec.weight.T.cpu().detach().numpy()
    phi = softmax_np(phi, -1)
    return phi


def get_topic_words(n_topic, model, voc, save_file=None, topn=10):
    topic_list = []
    phi = get_phi(model)
    idx2token = {index: word for index, word in enumerate(voc)}

    for i in range(n_topic):
        top_word_idx = np.argsort(phi[i, :])[::-1]
        top_word_idx = top_word_idx[0:topn]
        top_words = itemgetter(*top_word_idx)(idx2token)
        topic_list.append(top_words)

    if save_file:
        with open(save_file, 'w') as file:
            for item in topic_list:
                file.write(' '.join(item) + '\n')
            file.close()

    return topic_list


def get_theta(model, dataset, batch_size=200):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    theta_list = []
    for x, _ in data_loader:
        _, _, _, theta_batch, _ = model(x.float().cuda())
        theta_list.append(theta_batch)
    theta_torch = torch.cat(theta_list, axis=0).to(torch.float64)
    theta_torch = softmax(theta_torch, dim=1)

    return theta_torch.cpu().detach().numpy()


def perplexity(model, test_1, test_2, batch_size=200):
    data_loader = DataLoader(test_1, batch_size=batch_size, shuffle=False)
    model.eval()

    doc_word_list = []
    for x, _ in data_loader:
        _, _, _, _, doc_word_batch = model(x.float().cuda())
        doc_word_list.append(doc_word_batch)
    doc_word_torch = torch.cat(doc_word_list, axis=0).to(torch.float64)

    preds = doc_word_torch.detach().cpu().numpy()
    sums_2 = test_2.data.sum(1)
    recon_loss = -(preds * test_2.data).sum(1)
    loss = recon_loss / (sums_2 + 1e-10)
    loss = np.nanmean(loss)
    ppl = round(math.exp(loss), 1)

    return ppl


def get_doc_word(model, doc, voc, batch_size=200, topn=10):
    voc = {index: word for index, word in enumerate(voc)}
    data_loader = DataLoader(doc, batch_size=batch_size, shuffle=False)
    model.eval()

    doc_word_list = []
    for x, _ in data_loader:
        _, _, _, _, doc_word_batch = model(x.float().cuda())
        doc_word_list.append(doc_word_batch)
    doc_word_torch = torch.cat(doc_word_list, axis=0).to(torch.float64)
    doc_word = doc_word_torch.detach().cpu().numpy()

    # sort and keep only topn
    indices = np.argsort(-doc_word, axis=1)
    sorted_arr = np.take_along_axis(doc_word, indices, axis=1)
    top_n_sorted_arr = sorted_arr[:, :topn]
    top_n_indices = indices[:, :topn]

    top_words_list = []
    for i in range(top_n_indices.shape[0]):
        doc_top_words = itemgetter(*top_n_indices[i])(voc)
        top_words_list.append(doc_top_words)

    # normalise as distribution
    top_n_sorted_arr = top_n_sorted_arr / top_n_sorted_arr.sum(1)[:, np.newaxis]

    return top_words_list, top_n_sorted_arr