import argparse
import os
from gensim.models.wrappers import LdaMallet
from gensim.matutils import Sparse2Corpus
import sys
sys.path.append('../Topic_Model_Evaluation-main')
from read_data import *
from TM_eval import evaluation
import gensim.downloader as api

# set up mallet and set your own path here
# path_to_mallet_binary = "/home/<someone>/Mallet/bin/mallet"


parser = argparse.ArgumentParser(description='LDA')
parser.add_argument("--name", default='LDA', type=str)
parser.add_argument("--dataset", default='20News', type=str)
parser.add_argument("--n_topic", default=100, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--eval_step", default=5, type=int)
parser.add_argument("--alpha", help="doc_topic prior", default=50)
args = parser.parse_args()
args.alpha = 1/args.n_topic


def run_lda():
    # load data
    data_file = os.path.join('datasets', args.dataset, 'data.mat')
    data_dict = load_data(data_file, is_to_dense=False)
    train_data = sparse.csr_matrix(data_dict['train_data'])

    # convert sparse matrix to gensim corpus
    corpus_train = Sparse2Corpus(train_data.transpose())

    # convert voc list to dict
    voc = {k: v for k, v in enumerate(data_dict['voc'])}

    # get embed model for word similarity
    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')

    # train LDA
    for iteration in range(args.eval_step, args.epochs + args.eval_step, args.eval_step):
        lda_model = LdaMallet(path_to_mallet_binary,
                              corpus=corpus_train,
                              num_topics=args.n_topic,
                              alpha=args.alpha,
                              id2word=voc,
                              workers=10,
                              optimize_interval=50,
                              iterations=iteration,
                              topic_threshold=0.0,
                              random_seed=args.seed
                              )

        parameter_setting = '%s_%s_K%s_RS%s_epochs:%s' % (args.name, args.dataset, args.n_topic, args.seed, iteration)

        evaluation(lda_model, parameter_setting, data_dict, args, model_glove)


if __name__ == '__main__':
    run_lda()

