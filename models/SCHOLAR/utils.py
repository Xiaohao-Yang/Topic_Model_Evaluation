from torch.nn.functional import softmax
from operator import itemgetter
import os
import gensim
from scipy import io as sio
from sklearn.preprocessing import OneHotEncoder
import gensim.downloader as api
import math
from scipy import sparse
import torch
import sys
sys.path.append('../TM_Eval_Github')
from TM_eval import topic_diversity, text_clustering
from word_set_similarity import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix


def load_scholar_data(mat_file_name, is_to_dense=True):
    data = sio.loadmat(mat_file_name)
    train_data = data['wordsTrain'].transpose()
    test_data = data['wordsTest'].transpose()
    voc = data['vocabulary']
    voc = [v[0][0] for v in voc]

    word_embedding = []
    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')
    for v in voc:
        word_embedding.append(model_glove[v])
    word_embedding = np.array(word_embedding)

    test1 = data['test1'].transpose()
    test2 = data['test2'].transpose()
    train_label = data['labelsTrain']
    test_label = data['labelsTest']

    if is_to_dense:
        train_data = sparse2dense(train_data)
        test_data = sparse2dense(test_data)
        test1 = sparse2dense(test1)
        test2 = sparse2dense(test2)

    # load text data
    try:
        train_text, test_text = data['train_text'], data['test_text']
        train_text = [item[0][0].strip() for item in train_text]
    except:
        train_text, test_text = None, None

    # load llama output
    try:
        train_llama, test_llama = data['train_llama'], data['test_llama']
        train_llama = [item[0][0].strip() for item in train_llama]
        test_llama = [item[0][0].strip() for item in test_llama]
    except:
        train_llama, test_llama = None, None

    # change label to one-hot vector
    label_encoder = OneHotEncoder()
    train_label = label_encoder.fit_transform(train_label).todense()
    test_label_oneHot = label_encoder.fit_transform(test_label).todense()

    if not sparse.isspmatrix(train_data):
        train_data = sparse.csr_matrix(train_data).astype('float32')
        test_data = sparse.csr_matrix(test_data).astype('float32')

    data_dict = {
        'train_data': train_data,
        'train_label': train_label,
        'test_data': test_data,
        'test_label': test_label_oneHot,
        'test_label_normal': test_label,
        'word_embeddings': word_embedding,
        'voc': voc,
        'test1': test1,
        'test2': test2,
        'train_text': train_text,
        'test_text': test_text,
        'train_llama': train_llama,
        'test_llama': test_llama
    }

    return data_dict, model_glove


def get_doc_word_top(model, data, voc, topn=10):
    data = sparse2dense(data)
    _, doc_word = model.predict(data, None, None)
    doc_word = np.log(doc_word).astype(np.float64)

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


def perplexity(data_dict, model):
    test1 = sparse2dense(data_dict['test1'])
    test2 = sparse2dense(data_dict['test2'])

    _, preds = model.predict(test1, None, None)

    sums_2 = test2.sum(1)
    recon_loss = -(np.log(preds) * test2).sum(1)
    loss = recon_loss / sums_2
    loss = np.nanmean(loss)
    ppl = round(math.exp(loss), 1)

    return ppl


def print_topics(beta, voc, topn=10, save_file=None):
    K = beta.shape[0]
    topic_list = []
    for i in range(K):
        top_word_idx = np.argsort(beta[i, :])[::-1]
        top_word_idx = top_word_idx[0:topn]
        top_words = itemgetter(*top_word_idx)(voc)
        topic_list.append(top_words)

    if save_file:
        with open(save_file, 'w') as file:
            for item in topic_list:
                file.write(' '.join(item) + '\n')
            file.close()
    return topic_list


def rf_cls(train_theta, train_y, test_theta, test_y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=0)

    train_theta = train_theta.astype('float32')
    test_theta = test_theta.astype('float32')

    test_theta = np.nan_to_num(test_theta, posinf=3.4e+10, neginf=-3.4e+10)
    train_theta = np.nan_to_num(train_theta, posinf=3.4e+10, neginf=-3.4e+10)

    train_y = np.ravel(np.argmax(train_y, axis=1))
    test_y = np.ravel(np.argmax(test_y, axis=1))

    clf.fit(train_theta, train_y)
    predict_test = clf.predict(test_theta)
    acc = metrics.accuracy_score(test_y, predict_test)

    return acc


def evaluate_phase(options, ppl, epochs, data_dict, model, PC, TC, test_PC, test_TC):
    name = model._model.name
    parameter_setting = '%s_%s_K%s_RS%s_epochs:%s_LR%s' % (name, options.dataset, options.n_topic, options.seed, epochs, options.lr)
    print(parameter_setting)

    # save beta
    save_file_topics = None
    beta = model.get_weights()
    beta_torch = torch.tensor(beta)
    beta_normalised = softmax(beta_torch, -1).cpu().numpy()
    print_topics(beta, data_dict['voc'], save_file_topics)

    # topic diversity
    td = topic_diversity(beta_normalised)

    # doc classification evaluation
    inference_batch_size = 1000
    theta_train = save_document_representations(model, data_dict['train_data'], data_dict['train_label'], PC, TC, batch_size=inference_batch_size)
    theta_test = save_document_representations(model, data_dict['test_data'], data_dict['test_label'], test_PC, test_TC, batch_size=inference_batch_size)
    doc_acc = rf_cls(theta_train, data_dict['train_label'], theta_test, data_dict['test_label'])
    purity_test, nmi_test = text_clustering(theta_test, data_dict['test_label_normal'])

    #################### doc_words ####################
    ###################################################
    test_doc_top_words, dis1 = get_doc_word_top(model, data_dict['test_data'], data_dict['voc'])
    test_doc_top_words_llama = data_dict['test_llama']
    test_doc_top_words_llama = [item.split(' ') for item in test_doc_top_words_llama]
    dis2 = make_uni_dis(test_doc_top_words_llama)

    doc_word_sim_overlap = wordlists_similarity_overlap(test_doc_top_words, test_doc_top_words_llama,method='overlap')
    doc_word_sim_synset = wordlists_similarity_overlap(test_doc_top_words, test_doc_top_words_llama,method='synset')
    doc_word_sim_oa = wordlists_similarity_distance(test_doc_top_words, dis1, test_doc_top_words_llama, dis2,
                                               model.embedding_model, method='OA')
    doc_word_sim_ot = wordlists_similarity_distance(test_doc_top_words, dis1, test_doc_top_words_llama, dis2,
                                             model.embedding_model, method='OT')

    print('############################################')
    print('doc classification acc: ', doc_acc)
    print('test ppl: ', ppl)
    print('topic diversity: ', td)
    print('purity_test: ', purity_test)
    print('nmi_test: ', nmi_test)
    print('test doc words similarity overlap: ', doc_word_sim_overlap)
    print('test doc words similarity synset: ', doc_word_sim_synset)
    print('test doc words similarity OA: ', doc_word_sim_oa)
    print('test doc words similarity OT: ', doc_word_sim_ot)
    print('##########################################')


def save_document_representations(model, X, Y, PC, TC, batch_size=200):
    if Y is not None:
        Y = np.zeros_like(Y)

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    thetas = []

    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs = get_minibatch(X, Y, PC, TC, i, batch_size)
        thetas.append(model.compute_theta(batch_xs, batch_ys, batch_pcs, batch_tcs))
    theta = np.vstack(thetas)

    return theta


def train_dev_split(options, rng):
    # randomly split into train and dev
    if options.dev_folds > 0:
        n_dev = int(options.n_train / options.dev_folds)
        indices = np.array(range(options.n_train), dtype=int)
        rng.shuffle(indices)
        if options.dev_fold < options.dev_folds - 1:
            dev_indices = indices[n_dev * options.dev_fold: n_dev * (options.dev_fold +1)]
        else:
            dev_indices = indices[n_dev * options.dev_fold:]
        train_indices = list(set(indices) - set(dev_indices))
        return train_indices, dev_indices

    else:
        return None, None


def split_matrix(train_X, train_indices, dev_indices):
    # split a matrix (word counts, labels, or covariates), into train and dev
    if train_X is not None and dev_indices is not None:
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        return train_X, dev_X
    else:
        return train_X, None


def get_init_bg(data):
    #Compute the log background frequency of all words
    #sums = np.sum(data, axis=0)+1
    n_items, vocab_size = data.shape
    sums = np.array(data.sum(axis=0)).reshape((vocab_size,))+1.
    print("Computing background frequencies")
    print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg


def load_word_vectors(options, rng, vocab):
    # load word2vec vectors if given
    if options.word2vec_file is not None:
        vocab_size = len(vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        # randomly initialize word vectors for each term in the vocabualry
        embeddings = np.array(rng.rand(options.emb_dim, vocab_size) * 0.25 - 0.5, dtype=np.float32)
        count = 0
        print("Loading word vectors")
        # load the word2vec vectors
        pretrained = gensim.models.KeyedVectors.load_word2vec_format(options.word2vec_file, binary=True)

        # replace the randomly initialized vectors with the word2vec ones for any that are available
        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[:, index] = pretrained[word]

        print("Found embeddings for %d words" % count)
        update_embeddings = False
    else:
        embeddings = None
        update_embeddings = True

    return embeddings, update_embeddings


def make_network(options, vocab_size, word_embedding, label_type=None, n_labels=0, n_prior_covars=0, n_topic_covars=0):
    # Assemble the network configuration parameters into a dictionary
    network_architecture = \
        dict(embedding_dim=options.emb_dim,
             n_topics=options.n_topic,
             vocab_size=vocab_size,
             word_embedding=word_embedding,
             label_type=label_type,
             n_labels=n_labels,
             n_prior_covars=n_prior_covars,
             n_topic_covars=n_topic_covars,
             l1_beta_reg=options.l1_topics,
             l1_beta_c_reg=options.l1_topic_covars,
             l1_beta_ci_reg=options.l1_interactions,
             l2_prior_reg=options.l2_prior_covars,
             classifier_layers=1,
             use_interactions=options.interactions,
             dist=options.dist,
             model=options.model
             )
    return network_architecture


def create_minibatch(X, Y, PC, TC, batch_size=200, rng=None):
    # Yield a random minibatch
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)

        X_mb = np.array(X[ixs, :].todense()).astype('float32')
        if Y is not None:
            Y_mb = Y[ixs, :].astype('float32')
        else:
            Y_mb = None

        if PC is not None:
            PC_mb = PC[ixs, :].astype('float32')
        else:
            PC_mb = None

        if TC is not None:
            TC_mb = TC[ixs, :].astype('float32')
        else:
            TC_mb = None

        yield X_mb, Y_mb, PC_mb, TC_mb


def get_minibatch(X, Y, PC, TC, batch, batch_size=200):
    # Get a particular non-random segment of the data
    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / float(batch_size)))
    if batch < n_batches - 1:
        ixs = np.arange(batch * batch_size, (batch + 1) * batch_size)
    else:
        ixs = np.arange(batch * batch_size, n_items)

    if sparse.isspmatrix(X[ixs, :]):
        X_mb = np.array(X[ixs, :].todense()).astype('float32')
    else:
        X_mb = np.array(X[ixs, :]).astype('float32')

    if Y is not None:
        Y_mb = Y[ixs, :].astype('float32')
    else:
        Y_mb = None

    if PC is not None:
        PC_mb = PC[ixs, :].astype('float32')
    else:
        PC_mb = None

    if TC is not None:
        TC_mb = TC[ixs, :].astype('float32')
    else:
        TC_mb = None

    return X_mb, Y_mb, PC_mb, TC_mb


def predict_label_probs(model, X, PC, TC, batch_size=200, eta_bn_prop=0.0):
    # Predict a probability distribution over labels for each instance using the classifier part of the network

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    pred_probs_all = []

    # make predictions on minibatches and then combine
    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs = get_minibatch(X, None, PC, TC, i, batch_size)
        Z, pred_probs = model.predict(batch_xs, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop)
        pred_probs_all.append(pred_probs)

    pred_probs = np.vstack(pred_probs_all)

    return pred_probs





def print_top_words(beta, feature_names, topic_names=None, n_pos=8, n_neg=8, sparsity_threshold=1e-5, values=False):
    """
    Display the highest and lowest weighted words in each topic, along with mean ave weight and sparisty
    """
    sparsity_vals = []
    maw_vals = []
    for i in range(len(beta)):
        # sort the beta weights
        order = list(np.argsort(beta[i]))
        order.reverse()
        output = ''
        # get the top words
        for j in range(n_pos):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        order.reverse()
        if n_neg > 0:
            output += ' / '
        # get the bottom words
        for j in range(n_neg):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        # compute sparsity
        sparsity = float(np.sum(np.abs(beta[i]) < sparsity_threshold) / float(len(beta[i])))
        maw = np.mean(np.abs(beta[i]))
        sparsity_vals.append(sparsity)
        maw_vals.append(maw)
        output += '; sparsity=%0.4f' % sparsity

        # print the topic summary
        if topic_names is not None:
            output = topic_names[i] + ': ' + output
        else:
            output = str(i) + ': ' + output
        print(output)

    # return mean average weight and sparsity
    return np.mean(maw_vals), np.mean(sparsity_vals)


def print_top_bg(bg, feature_names, n_top_words=10):
    # Print the most highly weighted words in the background log frequency
    print('Background frequencies of top words:')
    print(" ".join([feature_names[j]
                    for j in bg.argsort()[:-n_top_words - 1:-1]]))
    temp = bg.copy()
    temp.sort()
    print(np.exp(temp[:-n_top_words-1:-1]))


def evaluate_perplexity(model, X, Y, PC, TC, batch_size, eta_bn_prop=0.0):
    # Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    n_items, vocab_size = X.shape
    doc_sums = np.array(X.sum(axis=1), dtype=float).reshape((n_items,))
    X = X.astype('float32')
    if Y is not None:
        Y = Y.astype('float32')
    if PC is not None:
        PC = PC.astype('float32')
    if TC is not None:
        TC = TC.astype('float32')
    losses = []

    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs = get_minibatch(X, Y, PC, TC, i, batch_size)
        batch_losses = model.get_losses(batch_xs, batch_ys, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop)
        losses.append(batch_losses)
    losses = np.hstack(losses)
    perplexity = np.exp(np.mean(losses / doc_sums))
    return perplexity


def print_topic_label_associations(options, label_names, model, n_prior_covars, n_topic_covars):
    # Print associations between topics and labels
    if options.n_labels > 0 and options.n_labels < 7:
        print("Label probabilities based on topics")
        print("Labels:", ' '.join([name for name in label_names]))
    probs_list = []
    for k in range(options.n_topics):
        Z = np.zeros([1, options.n_topics]).astype('float32')
        Z[0, k] = 1.0
        Y = None
        if n_prior_covars > 0:
            PC = np.zeros([1, n_prior_covars]).astype('float32')
        else:
            PC = None
        if n_topic_covars > 0:
            TC = np.zeros([1, n_topic_covars]).astype('float32')
        else:
            TC = None

        probs = model.predict_from_topics(Z, PC, TC)
        probs_list.append(probs)
        if options.n_labels > 0 and options.n_labels < 7:
            output = str(k) + ': '
            for i in range(options.n_labels):
                output += '%.4f ' % probs[0, i]
            print(output)

    probs = np.vstack(probs_list)
    np.savez(os.path.join(options.output_dir, 'topics_to_labels.npz'), probs=probs, label=label_names)