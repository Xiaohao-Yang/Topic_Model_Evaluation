import numpy as np
import scipy.io as sio
import pickle
import gensim.downloader as api
from scipy import sparse
import os
import torch


def get_topic_words(n_topic, model, voc, save_file=None, topn=10):
    with torch.no_grad():
        topics = []
        gammas = model.model.get_beta()

        for k in range(n_topic):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()
                             [-topn:][::-1])
            topic_words = [voc[a] for a in top_words]
            topics.append(topic_words)

    if save_file:
        with open(save_file, 'w') as file:
            for item in topics:
                file.write(' '.join(item) + '\n')
            file.close()

    return topics


def get_phi(model):
    phi = model.model.get_beta().cpu().detach().numpy()
    return phi


def get_theta(model, dataset):
    if dataset == 'train':
        theta = model.get_theta_train()
    else:
        theta = model.get_theta_test()
    return theta.cpu().detach().numpy()


def get_doc_word(model, topn=10):
    top_words, top_weight = model.get_doc_top_words(topn)
    return top_words, top_weight


def load_data_etm(dataset):
    with open('models/ETM/datasets/%s/voc.txt' % dataset, 'r') as file:
        voc = file.read().split(' ')
        file.close()

    word_embeddings = np.load('models/ETM/datasets/%s/word_embeddings.npy' % dataset)

    with open('models/ETM/datasets/%s/train.pkl' % dataset, 'rb') as file:
        train_data = pickle.load(file)
        file.close()
    with open('models/ETM/datasets/%s/test.pkl' % dataset, 'rb') as file:
        test_data = pickle.load(file)
        file.close()

    ################## load llama output ##################
    ######################################################
    llama_data_path = 'datasets/%s/data.mat' % dataset
    llama_data = sio.loadmat(llama_data_path)
    try:
        test_llama = llama_data['test_llama']
        test_llama = [item[0][0].strip() for item in test_llama]
    except:
        test_llama = None

    return train_data, test_data, word_embeddings, voc, test_llama


def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix


def format_data(dataset, etm_data_path):
    data = sio.loadmat(os.path.join('datasets', dataset, 'data.mat'))

    print('Formatting data for ETM...')

    train_data = data['wordsTrain'].transpose()
    test_data = data['wordsTest'].transpose()
    test1 = data['test1'].transpose()
    test2 = data['test2'].transpose()
    train_label = data['labelsTrain']
    test_label = data['labelsTest']

    train_data = sparse2dense(train_data)
    test_data = sparse2dense(test_data)
    test1_data = sparse2dense(test1)
    test2_data = sparse2dense(test2)

    voc = data['vocabulary']
    voc = [v[0][0] for v in voc]

    print('Loading word embedding model...')
    model_glove = api.load("glove-wiki-gigaword-300")
    print('Loading done!')
    word_embeddings = []
    for item in voc:
        word_embeddings.append(model_glove[item])
    word_embeddings = np.array(word_embeddings)

    train_data_dict = {'tokens': [],
                       'counts': [],
                       'labels': train_label}
    for i in range(train_data.shape[0]):
        token_idx = np.where(train_data[i] > 0)[0].astype('int32')
        counts = train_data[i][token_idx].astype('int64')
        train_data_dict['tokens'].append(token_idx)
        train_data_dict['counts'].append(counts)

    test_data_dict = {'tokens': [],
                      'counts': []}
    test1_data_dict = {'tokens': [],
                      'counts': []}
    test2_data_dict = {'tokens': [],
                      'counts': []}

    for i in range(test_data.shape[0]):
        # for original test
        token_idx = np.where(test_data[i] > 0)[0].astype('int32')
        counts = test_data[i][token_idx].astype('int64')
        test_data_dict['tokens'].append(token_idx)
        test_data_dict['counts'].append(counts)

        # for test1
        token_idx1 = np.where(test1_data[i] > 0)[0].astype('int32')
        counts1 = test1_data[i][token_idx1].astype('int64')
        test1_data_dict['tokens'].append(token_idx1)
        test1_data_dict['counts'].append(counts1)

        # for test2
        token_idx2 = np.where(test2_data[i] > 0)[0].astype('int32')
        counts2 = test2_data[i][token_idx2].astype('int64')
        test2_data_dict['tokens'].append(token_idx2)
        test2_data_dict['counts'].append(counts2)

    test_data_all = {'test':test_data_dict,
                     'test1':test1_data_dict,
                     'test2':test2_data_dict,
                     'labels': test_label}

    os.makedirs(etm_data_path)
    with open('%s/train.pkl' % etm_data_path, 'wb') as file:
        pickle.dump(train_data_dict, file)
        file.close()
    with open('%s/test.pkl' % etm_data_path, 'wb') as file:
        pickle.dump(test_data_all, file)
        file.close()
    with open('%s/voc.txt' % etm_data_path, 'w') as file:
        file.write(' '.join(voc))
        file.close()
    np.save('%s/word_embeddings' % etm_data_path, word_embeddings)

    print('Data formatted for ETM!')