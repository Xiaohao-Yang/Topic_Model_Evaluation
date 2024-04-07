from tqdm import tqdm
from model import *
import argparse
import torch
import math
import numpy as np
import sys
sys.path.append('../Topic_Model_Evaluation-main')
from read_data import *
from TM_eval import evaluation
import gensim.downloader as api
from utils import batch_indices, get_voc_embeddings
import os


parser = argparse.ArgumentParser(description='NSTM')
parser.add_argument("--name", type=str, default="NSTM")
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--dataset", type=str, default='20News')
parser.add_argument("--seed", help="Random seed", default=1, type=int)
parser.add_argument("--n_topic", default=50, type=int)
parser.add_argument("--eval_step", default=5, type=int)
parser.add_argument("--lr", help="Learning rate", default=0.001, type=float)
parser.add_argument("--rec_loss_weight", default=0.07, type=float)

parser.add_argument("--hidden_dim", help="Hidden dimension", default=200, type=int)
parser.add_argument("--batch_size", default=200, type=int)
parser.add_argument("--sh_iterations", default=50, type=int)
parser.add_argument("--sh_epsilon", default=0.001, type=float)
parser.add_argument("--sh_alpha", default=20, type=int)
args = parser.parse_args()


def run_ntsm():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    data_dir = os.path.join('datasets', args.dataset, 'data.mat')
    data_dict = load_data(data_dir)

    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')
    word_embeddings = get_voc_embeddings(data_dict['voc'], model_glove)

    V = data_dict['train_data'].shape[1]       # voc size
    N = data_dict['train_data'].shape[0]       # train size
    L = word_embeddings.shape[1]               # embedding dim

    # word embedding
    word_embedding = torch.tensor(word_embeddings, dtype=torch.float32, device=device, requires_grad=True)

    # topic embedding
    topic_embedding = torch.zeros(size=(args.n_topic, L),dtype=torch.float32, requires_grad=True, device=device)
    torch.nn.init.trunc_normal_(topic_embedding, std=0.1)

    # set model
    model = encoder(V, args.hidden_dim, args.n_topic).to(device)
    loss_function = myLoss()
    # optimizer
    optimize_params = list(model.parameters())
    optimize_params.append(topic_embedding)
    optimizer = torch.optim.Adam(optimize_params, lr=args.lr, betas=(0.99, 0.999))

    # train loop
    nb_batches = int(math.ceil(float(N) / args.batch_size))
    assert nb_batches * args.batch_size >= N
    for epoch in range(args.epochs):
        idxlist = np.random.permutation(N)  # can be used to index train_text
        rec_loss_avg, sh_loss_avg, joint_loss_avg = 0., 0., 0.

        for batch in tqdm(range(nb_batches)):
            optimizer.zero_grad()
            start, end = batch_indices(batch, N, args.batch_size)
            X = data_dict['train_data'][idxlist[start:end]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = torch.tensor(X, device=device)

            doc_word = F.softmax(X, dim=1)
            doc_topic = model(X)

            word_embedding_norm = F.normalize(word_embedding, p=2, dim=1)
            topic_embedding_norm = F.normalize(topic_embedding, p=2, dim=1)
            topic_word = torch.matmul(topic_embedding_norm, word_embedding_norm.t())

            M = 1 - topic_word

            rec_loss_batch, sh_rec_loss_batch, joint_loss_batch = \
                loss_function(X, doc_topic, doc_word, M, topic_embedding, args.sh_alpha, args.rec_loss_weight)

            joint_loss_batch.backward()
            optimizer.step()

            rec_loss_avg += rec_loss_batch.item()
            sh_loss_avg += sh_rec_loss_batch.item()
            joint_loss_avg += joint_loss_batch.item()

        if (epoch+1) % args.eval_step == 0:
            parameter_setting = ('NSTM_%s_K%s_RS%s_LR%s_RecW%s' %
                                 (args.dataset, args.n_topic, args.seed, args.lr, args.rec_loss_weight))
            data_dict['topic_word'] = topic_word.clone().detach().cpu().numpy()
            evaluation(model, parameter_setting, data_dict, args, model_glove)
            model.train()


if __name__ == '__main__':
    run_ntsm()
