from scipy.special import softmax as softmax_np
from torch.nn.functional import softmax
import math
import numpy as np
from operator import itemgetter
import torch
from torch.utils.data import DataLoader


def get_phi(model):
    phi = model.beta.cpu().detach().numpy()
    phi = softmax_np(phi, -1)
    return phi


def get_topic_words(n_topic, model, voc, save_file, topn=10):
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


def predict(model, dataset, batch_size=200, topn=10):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    doc_word_dist = []
    with torch.no_grad():
        for batch_samples in loader:
            # batch_size x vocab_size
            X = batch_samples['X'].cuda()
            model.zero_grad()
            _, _, _, _, _, word_dists, _ = model(X)

            sorted_weight, indices = torch.sort(word_dists, dim=1, descending=True)
            word_dists = word_dists.cpu().numpy()
            doc_word_dist.append(word_dists)
            preds += [indices[:, :topn]]

        preds = torch.cat(preds, dim=0)
        doc_word_dist_np = np.concatenate(doc_word_dist, axis=0)

    return preds, doc_word_dist_np


def get_theta(model,  dataset, batch_size=200):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    theta_list = []
    with torch.no_grad():
        for batch_samples in loader:
            X = batch_samples['X'].cuda()
            model.zero_grad()
            _, _, _, _, _, _, theta_batch = model(X)
            theta_list.append(theta_batch)

        theta = torch.cat(theta_list, dim=0).to(torch.float64)
        theta = softmax(theta, dim=1)

    return theta.cpu().detach().numpy()


def get_doc_word(model, dataset, voc, batch_size=200, topn=10):
    doc_word_test, doc_word_mass = predict(model, dataset, batch_size, topn)

    top_words_list = []
    for i in range(doc_word_test.shape[0]):
        doc_top_words = itemgetter(*doc_word_test[i])(voc)
        top_words_list.append(doc_top_words)

    # get top doc-word mass
    sorted_indices = np.argsort(-doc_word_mass, axis=1)
    top_doc_word_dis = doc_word_mass[np.arange(doc_word_mass.shape[0])[:, None], sorted_indices[:, :topn]].astype(np.float64)
    # normalise as distribution
    top_doc_word_dis = top_doc_word_dis / top_doc_word_dis.sum(1)[:, np.newaxis]

    return top_words_list, top_doc_word_dis


def perplexity(model, test_1, test_2):
    _, doc_word_dist = predict(model, test_1)
    preds = np.log(doc_word_dist)
    sums_2 = test_2.X.sum(1)

    recon_loss = -(preds * test_2.X).sum(1)
    loss = (recon_loss / sums_2 + 1e-10)
    loss = np.nanmean(loss)
    ppl = round(math.exp(loss), 1)

    return ppl