import os
import argparse
from pytorchavitm import AVITM
from pytorchavitm.datasets import BOWDataset
import gensim.downloader as api
import sys
sys.path.append('../Topic_Model_Evaluation-main')
from read_data import *
import torch
import numpy as np


parser = argparse.ArgumentParser(description='ProdLDA')
parser.add_argument("--name", type=str, default="PLDA")
parser.add_argument('--dataset', type=str, default='20News')
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument("--eval_step", default=5, type=int)
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def main():
    # load data
    data_file = os.path.join('datasets', args.dataset, 'data.mat')
    data_dict = load_data(data_file)

    # voc
    voc_size = len(data_dict['voc'])
    idx2token = {index: word for index, word in enumerate(data_dict['voc'])}

    # create dataset
    train_data = BOWDataset(data_dict['train_data'], idx2token)
    test_data = BOWDataset(data_dict['test_data'], idx2token)
    test1_data = BOWDataset(data_dict['test1'], idx2token)
    test2_data = BOWDataset(data_dict['test2'], idx2token)

    # update data dict
    data_dict['train_data'] = train_data
    data_dict['test_data'] = test_data
    data_dict['test1'] = test1_data
    data_dict['test2'] = test2_data

    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')

    avitm = AVITM(input_size=voc_size, n_components=args.n_topic, lr=args.lr, num_epochs=args.epochs,
                  hidden_sizes=(200, ), batch_size=200, momentum=0.9)

    avitm.fit(data_dict, model_glove, args)


if __name__ == '__main__':
    main()





