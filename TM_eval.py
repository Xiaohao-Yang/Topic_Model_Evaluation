from word_set_similarity import *
from models.model_utils import model_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics


def text_clustering(data, labels_true, num_clusters=30):
    # standardization
    mu = np.mean(data, axis=-1, keepdims=True)
    sigma = np.std(data, axis=-1, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1)
    data = (data - mu) / sigma

    # clustering based on Euclidean distance
    estimator = KMeans(n_clusters=num_clusters, random_state=0)
    estimator.fit(data)
    labels_pred = estimator.labels_

    labels_true = labels_true.reshape(-1,).astype('int64')
    purity_score = purity(labels_true, labels_pred)
    nmi_score = normalized_mutual_info_score(labels_true, labels_pred)

    return purity_score, nmi_score


def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    counts = []
    for c in clusters:
        indices = np.where(labels_pred == c)[0]
        max_votes = np.bincount(labels_true[indices]).max()
        counts.append(max_votes)
    return sum(counts) / labels_true.shape[0]


def rf_cls(train_theta, train_y, test_theta, test_y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=0)

    train_theta = train_theta.astype('float32')
    test_theta = test_theta.astype('float32')
    train_y = train_y.ravel()
    test_y = test_y.ravel()

    clf.fit(train_theta, train_y)
    predict_test = clf.predict(test_theta)
    acc = metrics.accuracy_score(test_y, predict_test)

    return acc


def topic_diversity(topic_matrix, top_k=25):
    num_topics = topic_matrix.shape[0]
    top_words_idx = np.zeros((num_topics, top_k))
    for k in range(num_topics):
        idx = np.argsort(topic_matrix[k, :])[::-1][:top_k]
        top_words_idx[k, :] = idx
    num_unique = len(np.unique(top_words_idx))
    num_total = num_topics * top_k
    td = num_unique / num_total

    return td


def evaluation(model, parameter_setting, data_dict, args, embedding_model):
    try:
        model.eval()
    except:
        pass
    utils = model_utils(args.name).funcs

    print('##########################################')
    print('Evaluation Phase ...')
    print(parameter_setting)

    # save topics
    # define your own path to save topics to file
    save_file_topics = None
    if args.name == 'NSTM':
        topic_top_words = utils.get_topic_words(data_dict['topic_word'], data_dict['voc'], save_file_topics, topn=10)
    else:
        topic_top_words = utils.get_topic_words(args.n_topic, model, data_dict['voc'], save_file_topics, topn=10)

    # get topic-word distribution
    if args.name == 'NSTM':
        phi = data_dict['topic_word']
    else:
        phi = utils.get_phi(model)

    # topic diversity
    td = topic_diversity(phi)

    # train theta and test theta
    train_theta = utils.get_theta(model, data_dict['train_data'])
    test_theta = utils.get_theta(model, data_dict['test_data'])

    # document completion perplexity
    if args.name == 'ETM':
        ppl_doc_completion = model._perplexity(data_dict['test_data'])
    elif args.name == 'NSTM':
        ppl_doc_completion = utils.perplexity(model, data_dict['test1'], data_dict['test2'], phi)
    else:
        ppl_doc_completion = utils.perplexity(model, data_dict['test1'], data_dict['test2'])

    # document classification ACC
    cls_acc = rf_cls(train_theta, data_dict['train_label'], test_theta, data_dict['test_label'])

    # k-means clustering
    purity_test, nmi_test = text_clustering(test_theta, data_dict['test_label'])

    # doc_words and probability mass from topic model
    if args.name in ['LDA', 'NSTM']:
        test_doc_top_words, test_doc_top_dis = utils.get_doc_word(phi, test_theta, data_dict['voc'], topn=10)
    elif args.name == 'ETM':
        test_doc_top_words, test_doc_top_dis = utils.get_doc_word(model, topn=10)
    else:
        test_doc_top_words, test_doc_top_dis = utils.get_doc_word(model, data_dict['test_data'], data_dict['voc'], topn=10)

    # doc_words and probability mass from LLM
    test_doc_top_words_llama = data_dict['test_llama']
    test_doc_top_words_llama = [item.split(' ') for item in test_doc_top_words_llama]
    test_doc_top_dis_llama = make_uni_dis(test_doc_top_words_llama)

    # compute agreements between words from topic model and llm
    doc_word_sim_overlap = wordlists_similarity_overlap(test_doc_top_words, test_doc_top_words_llama, method='overlap')
    doc_word_sim_synset = wordlists_similarity_overlap(test_doc_top_words, test_doc_top_words_llama, method='synset')
    doc_word_sim_oa = wordlists_similarity_distance(test_doc_top_words, test_doc_top_dis, test_doc_top_words_llama,
                                                    test_doc_top_dis_llama, embedding_model, method='OA')
    doc_word_sim_ot = wordlists_similarity_distance(test_doc_top_words, test_doc_top_dis, test_doc_top_words_llama,
                                                    test_doc_top_dis_llama, embedding_model, method='OT')

    ###########################################################################################
    ###########################################################################################
    print('############################################')
    print('doc classification acc: ', cls_acc)
    print('test ppl: ', ppl_doc_completion)
    print('topic diversity: ', td)
    print('purity_test: ', purity_test)
    print('nmi_test: ', nmi_test)
    print('test doc words similarity overlap: ', doc_word_sim_overlap)
    print('test doc words similarity synset: ', doc_word_sim_synset)
    print('test doc words similarity OA: ', doc_word_sim_oa)
    print('test doc words similarity OT: ', doc_word_sim_ot)
    print('##########################################')