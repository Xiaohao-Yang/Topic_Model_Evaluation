from optparse import OptionParser
from scholar import Scholar
from utils import *
import torch
import os
import sys
import time
import numpy as np


args = sys.argv[1:]
usage = "%prog input_dir"
parser = OptionParser(usage=usage)
parser.add_option('--n_topic', type=int, default=50)
parser.add_option('--lr', type=float, default=0.002)
parser.add_option('--model', type=str, default='SCHOLAR')
parser.add_option('--dataset', type=str, default='20News')
parser.add_option('--eval_step', type=int, default=5)
parser.add_option('--epochs', type=int, default=500)
parser.add_option('--seed', type=int, default=1, help='Random seed: default=%default')


parser.add_option('--topk', type=int, default=15)
parser.add_option('-m', dest='momentum', type=float, default=0.99,
                  help='beta1 for Adam: default=%default')
parser.add_option('--batch-size', dest='batch_size', type=int, default=200,
                  help='Size of minibatches: default=%default')
parser.add_option('--train-prefix', type=str, default='train',
                  help='Prefix of train set: default=%default')
parser.add_option('--test-prefix', type=str, default='test',
                  help='Prefix of test set: default=%default')
parser.add_option('--labels', type=str, default=False,
                  help='Read labels from input_dir/[train|test].labels.csv: default=%default')
parser.add_option('--prior-covars', type=str, default=None,
                  help='Read prior covariates from files with these names (comma-separated): default=%default')
parser.add_option('--topic-covars', type=str, default=None,
                  help='Read topic covariates from files with these names (comma-separated): default=%default')
parser.add_option('--interactions', action="store_true", default=False,
                  help='Use interactions between topics and topic covariates: default=%default')
parser.add_option('--covars-predict', action="store_true", default=False,
                  help='Use covariates as input to classifier: default=%default')
parser.add_option('--min-prior-covar-count', type=int, default=None,
                  help='Drop prior covariates with less than this many non-zero values in the training dataa: default=%default')
parser.add_option('--min-topic-covar-count', type=int, default=None,
                  help='Drop topic covariates with less than this many non-zero values in the training dataa: default=%default')
parser.add_option('-r', action="store_true", default=False,
                  help='Use default regularization: default=%default')
parser.add_option('--l1-topics', type=float, default=0.0,
                  help='Regularization strength on topic weights: default=%default')
parser.add_option('--l1-topic-covars', type=float, default=0.0,
                  help='Regularization strength on topic covariate weights: default=%default')
parser.add_option('--l1-interactions', type=float, default=0.0,
                  help='Regularization strength on topic covariate interaction weights: default=%default')
parser.add_option('--l2-prior-covars', type=float, default=0.0,
                  help='Regularization strength on prior covariate weights: default=%default')
parser.add_option('--o', dest='output_dir', type=str, default='output',
                  help='Output directory: default=%default')
parser.add_option('--emb-dim', type=int, default=300,
                  help='Dimension of input embeddings: default=%default')
parser.add_option('--w2v', dest='word2vec_file', type=str, default=None,
                  help='Use this word2vec .bin file to initialize and fix embeddings: default=%default')
parser.add_option('--alpha', type=float, default=1.0,
                  help='Hyperparameter for logistic normal prior: default=%default')
parser.add_option('--no-bg', action="store_true", default=False,
                  help='Do not use background freq: default=%default')
parser.add_option('--dev-folds', type=int, default=0,
                  help='Number of dev folds: default=%default')
parser.add_option('--dev-fold', type=int, default=0,
                  help='Fold to use as dev (if dev_folds > 0): default=%default')
parser.add_option('--device', type=int, default=0,
                  help='GPU to use: default=%default')
parser.add_option('--dist', type=int, default=0, help='distance')
options, args = parser.parse_args(args)

torch.manual_seed(options.seed)
np.random.seed(options.seed)


def train(model, network_architecture, data_dict, PC, TC, options, test_prior_covars,
          test_topic_covars, batch_size=200, training_epochs=100, init_eta_bn_prop=1.0, rng=None,
          bn_anneal=True, min_weights_sq=1e-7):

    # Train the model
    n_train, vocab_size = data_dict['train_data'].shape
    mb_gen = create_minibatch(data_dict['train_data'], data_dict['train_label'], PC, TC, batch_size=batch_size, rng=rng)
    total_batch = int(n_train / batch_size)
    batches = 0
    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon

    model.train()

    n_topics = network_architecture['n_topics']
    n_topic_covars = network_architecture['n_topic_covars']
    vocab_size = network_architecture['vocab_size']

    # create matrices to track the current estimates of the priors on the individual weights
    if network_architecture['l1_beta_reg'] > 0:
        l1_beta = 0.5 * np.ones([vocab_size, n_topics], dtype=np.float32) / float(n_train)
    else:
        l1_beta = None
    if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
        l1_beta_c = 0.5 * np.ones([vocab_size, n_topic_covars], dtype=np.float32) / float(n_train)
    else:
        l1_beta_c = None
    if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and network_architecture['use_interactions']:
        l1_beta_ci = 0.5 * np.ones([vocab_size, n_topics * n_topic_covars], dtype=np.float32) / float(n_train)
    else:
        l1_beta_ci = None

    # Training cycle
    for epoch in range(training_epochs):
        epoch_start_time = time.time()

        avg_cost = 0.
        avg_nl = 0.
        avg_kld = 0.
        avg_cl = 0.

        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_xs, batch_ys, batch_pcs, batch_tcs = next(mb_gen)

            # do one minibatch update
            cost, recon_y, thetas, nl, kld, cl = model.fit(batch_xs, batch_ys, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop,
                                                           l1_beta=l1_beta, l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci, current_model=options.model)

            # Compute average loss
            avg_cost += float(cost) / n_train * batch_size
            avg_nl += float(nl) / n_train * batch_size
            avg_kld += float(kld) / n_train * batch_size
            avg_cl += float(cl) / batch_size

            batches += 1
            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                sys.exit()

        # compute ppl every epochs
        model.eval()
        ppl = perplexity(data_dict, model)
        model.train()

        meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch+1, time.time() - epoch_start_time)
        print(meta + "| train loss {:5.2f} | (nll {:4.2f} | kld {:5.2f} | contrastive loss {:5.2f} | PPL {:5.2f})"
              .format(avg_cost, avg_nl, avg_kld, avg_cl, ppl))

        ##########################################################################
        ##########################################################################
        ##########################################################################
        ##########################################################################
        # if we're using regularization, update the priors on the individual weights
        if network_architecture['l1_beta_reg'] > 0:
            weights = model.get_weights().T
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta = 0.5 / weights_sq / float(n_train)

        if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
            weights = model.get_covar_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_c = 0.5 / weights_sq / float(n_train)

        if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and network_architecture['use_interactions']:
            weights = model.get_covar_interaction_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_ci = 0.5 / weights_sq / float(n_train)

        # anneal eta_bn_prop from 1.0 to 0.0 over training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(0.75 * training_epochs)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

        if (epoch + 1) % options.eval_step == 0:
            model.eval()
            evaluate_phase(options, ppl, epoch+1, data_dict, model, PC, TC, test_prior_covars, test_topic_covars)
            model.train()

    # finish training
    model.eval()
    return model


def main(data_dict, model_glove):
    if options.r:
        options.l1_topics = 1.0
        options.l1_topic_covars = 1.0
        options.l1_interactions = 1.0
    if options.seed is not None:
        rng = np.random.RandomState(options.seed)
        seed = options.seed
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))
        seed = None

    n_labels = data_dict['train_label'].shape[1]
    options.n_train, vocab_size = data_dict['train_data'].shape
    options.n_labels = n_labels

    # unused variables in original implementation
    train_prior_covars, prior_covar_selector, prior_covar_names, n_prior_covars = None, None, None, 0
    train_topic_covars, topic_covar_selector, topic_covar_names, n_topic_covars = None, None, None, 0
    test_prior_covars = None
    test_topic_covars = None
    label_type = None

    train_indices, dev_indices = train_dev_split(options, rng)
    train_X, dev_X = split_matrix(data_dict['train_data'], train_indices, dev_indices)
    train_prior_covars, dev_prior_covars = split_matrix(train_prior_covars, train_indices, dev_indices)
    train_topic_covars, dev_topic_covars = split_matrix(train_topic_covars, train_indices, dev_indices)
    n_train, _ = train_X.shape
    # initialize the background using overall word frequencies
    init_bg = get_init_bg(train_X)
    if options.no_bg:
        init_bg = np.zeros_like(init_bg)
    # load word vectors
    embeddings, update_embeddings = load_word_vectors(options, rng, data_dict['voc'])
    #################################################################################
    #################################################################################

    network_architecture = make_network(options, vocab_size, data_dict['word_embeddings'], label_type, n_labels, n_prior_covars,
                                        n_topic_covars)
    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ':', val)

    # create the model
    model = Scholar(network_architecture, model_glove, alpha=options.alpha, learning_rate=options.lr,
                    init_embeddings=embeddings, update_embeddings=update_embeddings, init_bg=init_bg,
                    adam_beta1=options.momentum, device=options.device, seed=seed,
                    classify_from_covars=options.covars_predict, model=options.model, topk=options.topk)

    # train the model
    print("Optimizing full model")
    model = train(model, network_architecture, data_dict, train_prior_covars, train_topic_covars, options,
                  test_prior_covars, test_topic_covars, training_epochs=options.epochs,
                  batch_size=options.batch_size, rng=rng)


if __name__ == '__main__':
    data_file = os.path.join('datasets', options.dataset, 'data.mat')
    data_dict, model_glove = load_scholar_data(data_file, False)
    main(data_dict, model_glove)







