import os.path
import argparse
import datetime
from nvdm_torch import *
import sys
sys.path.append('../TM_Eval_Github')
from read_data import *
from TM_eval import evaluation
import numpy as np
import gensim.downloader as api
from torch.utils.data import DataLoader
from dataset import *


parser = argparse.ArgumentParser(description='NVDM')
parser.add_argument("--name", type=str, default="NVDM")
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--eval_step', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)

parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--n_hidden', default=100, type=int)
parser.add_argument('--n_sample', default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    data_file = os.path.join('datasets', args.dataset, 'data.mat')
    data_dict = load_data(data_file)

    # create dataset object
    train_dataset = FeatDataset(data_dict['train_data'])
    test_dataset = FeatDataset(data_dict['test_data'])
    test1_data = FeatDataset(data_dict['test1'])
    test2_data = FeatDataset(data_dict['test2'])
    # update data dictionary
    data_dict['train_data'] = train_dataset
    data_dict['test_data'] = test_dataset
    data_dict['test1'] = test1_data
    data_dict['test2'] = test2_data

    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')

    # create model
    model = NVDM(len(data_dict['voc']), args.n_hidden, args.n_topic, args.n_sample).to(device)

    # data loader
    dataloader = DataLoader(data_dict['train_data'], batch_size=args.batch_size, shuffle=True)

    # record value
    loss_sum = 0.0
    ppx_sum = 0.0
    kld_sum = 0.0
    word_count = 0
    doc_count = 0

    # optimiser
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))

    # training loop
    for epoch in range(args.epochs):
        s = datetime.datetime.now()
        for data_batch, count_batch in dataloader:
            data_batch = data_batch.float().cuda()
            count_batch = count_batch.cuda()

            kld, recons_loss, _, _, _ = model(data_batch)
            # compute loss
            loss = kld + recons_loss

            loss_sum += torch.sum(loss).item()
            kld_sum += torch.mean(kld).item()
            word_count += torch.sum(count_batch).item()
            count_batch = torch.add(count_batch, 1e-12)
            ppx_sum += torch.sum(torch.div(loss, count_batch)).item()
            doc_count += len(data_batch)

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

        e = datetime.datetime.now()
        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_kld = kld_sum / len(dataloader)
        print('| Time : {} |'.format(e - s),
              '| Epoch train: {:d} |'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
              '| KLD: {:.5}'.format(print_kld))

        # evaluation phase
        if (epoch + 1) % args.eval_step == 0:
            parameter_setting = 'NVDM_%s_K%s_RS%s_epochs:%s_LR%s' % (args.dataset, args.n_topic, args.seed, epoch+1, args.lr)
            evaluation(model, parameter_setting, data_dict, args, model_glove)


if __name__ == '__main__':
    train()
