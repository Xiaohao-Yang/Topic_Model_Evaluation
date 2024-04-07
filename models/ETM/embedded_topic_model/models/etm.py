from __future__ import print_function
import torch
import numpy as np
import os
import math
from torch import optim
from gensim.models import KeyedVectors
from embedded_topic_model.models.model import Model
from embedded_topic_model.utils import data
from embedded_topic_model.utils import embedding
import sys
sys.path.append('../Topic_Model_Evaluation-main')
from TM_eval import evaluation
from operator import itemgetter


class ETM(object):
    def __init__(
        self,
        vocabulary,
        embeddings=None,
        use_c_format_w2vec=False,
        model_path=None,
        batch_size=1000,
        num_topics=50,
        rho_size=300,
        emb_size=300,
        t_hidden_size=800,
        theta_act='relu',
        train_embeddings=False,
        lr=0.005,
        lr_factor=4.0,
        epochs=20,
        optimizer_type='adam',
        seed=2019,
        enc_drop=0.0,
        clip=0.0,
        nonmono=10,
        wdecay=1.2e-6,
        anneal_lr=False,
        bow_norm=True,
        num_words=10,
        log_interval=2,
        visualize_every=50,
        eval_batch_size=1000,
        eval_perplexity=True,
        debug_mode=False,
        test_llama=None,
        embedding_model=None
    ):
        self.test_llama = test_llama
        self.embedding_model = embedding_model
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_topics = num_topics
        self.rho_size = rho_size
        self.emb_size = emb_size
        self.t_hidden_size = t_hidden_size
        self.theta_act = theta_act
        self.lr_factor = lr_factor
        self.epochs = epochs
        self.seed = seed
        self.enc_drop = enc_drop
        self.clip = clip
        self.nonmono = nonmono
        self.anneal_lr = anneal_lr
        self.bow_norm = bow_norm
        self.num_words = num_words
        self.log_interval = log_interval
        self.visualize_every = visualize_every
        self.eval_batch_size = eval_batch_size
        self.eval_perplexity = eval_perplexity
        self.debug_mode = debug_mode
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        self.embeddings = None if train_embeddings else self._initialize_embeddings(
            embeddings, use_c_format_w2vec=use_c_format_w2vec)

        self.model = Model(
            self.device,
            self.num_topics,
            self.vocabulary_size,
            self.t_hidden_size,
            self.rho_size,
            self.emb_size,
            self.theta_act,
            self.embeddings,
            train_embeddings,
            self.enc_drop,
            self.debug_mode).to(
            self.device)
        self.optimizer = self._get_optimizer(optimizer_type, lr, wdecay)

    def __str__(self):
        return f'{self.model}'

    def _get_extension(self, path):
        assert isinstance(path, str), 'path extension is not str'
        filename = path.split(os.path.sep)[-1]
        return filename.split('.')[-1]

    def _get_embeddings_from_original_word2vec(self, embeddings_file):
        if self._get_extension(embeddings_file) == 'txt':
            if self.debug_mode:
                print('Reading embeddings from original word2vec TXT file...')
            vectors = {}
            iterator = embedding.MemoryFriendlyFileIterator(embeddings_file)
            for line in iterator:
                word = line[0]
                if word in self.vocabulary:
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
            return vectors
        elif self._get_extension(embeddings_file) == 'bin':
            if self.debug_mode:
                print('Reading embeddings from original word2vec BIN file...')
            return KeyedVectors.load_word2vec_format(
                embeddings_file, 
                binary=True
            )
        else:
            raise Exception('Original Word2Vec file without BIN/TXT extension')

    def _initialize_embeddings(
        self, 
        embeddings,
        use_c_format_w2vec=False
    ):
        return torch.from_numpy(embeddings).to(self.device)


    def _get_optimizer(self, optimizer_type, learning_rate, wdecay):
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'adadelta':
            return optim.Adadelta(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=wdecay)
        elif optimizer_type == 'asgd':
            return optim.ASGD(
                self.model.parameters(),
                lr=learning_rate,
                t0=0,
                lambd=0.,
                weight_decay=wdecay)
        else:
            if self.debug_mode:
                print('Defaulting to vanilla SGD')
            return optim.SGD(self.model.parameters(), lr=learning_rate)


    def  _set_training_data(self, train_data):
        self.train_tokens = train_data['tokens']
        self.train_counts = train_data['counts']
        self.num_docs_train = len(self.train_tokens)
        self.train_labels = train_data['labels']


    def _set_test_data(self, test_data):
        self.test_tokens = test_data['test']['tokens']
        self.test_counts = test_data['test']['counts']
        self.num_docs_test = len(self.test_tokens)
        self.test_1_tokens = test_data['test1']['tokens']
        self.test_1_counts = test_data['test1']['counts']
        self.num_docs_test_1 = len(self.test_1_tokens)
        self.test_2_tokens = test_data['test2']['tokens']
        self.test_2_counts = test_data['test2']['counts']
        self.num_docs_test_2 = len(self.test_2_tokens)
        self.test_labels = test_data['labels']


    def _train(self, epoch):
        self.model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        indices = torch.randperm(self.num_docs_train)
        indices = torch.split(indices, self.batch_size)

        for idx, ind in enumerate(indices):
            self.optimizer.zero_grad()
            self.model.zero_grad()

            data_batch = data.get_batch(
                self.train_tokens,
                self.train_counts,
                ind,
                self.vocabulary_size,
                self.device)

            sums = data_batch.sum(1).unsqueeze(1)
            if self.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch

            # reconstruction loss and kl loss
            recon_loss, kld_theta, _ = self.model(
                data_batch, normalized_data_batch)

            total_loss = recon_loss + kld_theta

            total_loss.backward()

            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip)
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1

            if idx % self.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)

        if self.debug_mode:
            print('Epoch {} - Learning Rate: {} - KL theta: {} - Rec loss: {} - NELBO: {}'.format(
                epoch, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

    def _perplexity(self, test_data) -> float:
        self._set_test_data(test_data)

        self.model.eval()
        with torch.no_grad():
            # get \beta here
            beta = self.model.get_beta()

            # do dc here
            acc_loss = 0
            cnt = 0
            indices_1 = torch.split(
                torch.tensor(
                    range(
                        self.num_docs_test_1)),
                self.eval_batch_size)
            for idx, ind in enumerate(indices_1):
                # get theta from first half of docs
                data_batch_1 = data.get_batch(
                    self.test_1_tokens,
                    self.test_1_counts,
                    ind,
                    self.vocabulary_size,
                    self.device)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if self.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = self.model.get_theta(normalized_data_batch_1)

                # get prediction loss using second half
                data_batch_2 = data.get_batch(
                    self.test_2_tokens,
                    self.test_2_counts,
                    ind,
                    self.vocabulary_size,
                    self.device)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                recon_loss = -(preds * data_batch_2).sum(1)

                loss = recon_loss / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1

            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)

            return ppl_dc


    def fit(self, train_data, test_data, args):
        self._set_training_data(train_data)
        best_val_ppl = 1e9
        all_val_ppls = []

        for epoch in range(self.epochs):
            self._train(epoch)

            if self.eval_perplexity:

                val_ppl = self._perplexity(
                    test_data)
                if val_ppl < best_val_ppl:
                    if self.model_path is not None:
                        self._save_model(self.model_path)
                    best_val_ppl = val_ppl
                else:
                    # check whether to anneal lr
                    lr = self.optimizer.param_groups[0]['lr']
                    if self.anneal_lr and (len(all_val_ppls) > self.nonmono and val_ppl > min(
                            all_val_ppls[:-self.nonmono]) and lr > 1e-5):
                        self.optimizer.param_groups[0]['lr'] /= self.lr_factor

                all_val_ppls.append(val_ppl)

            if (epoch + 1) % args.eval_step == 0:
                self.model.eval()
                parameter_setting = 'ETM_%s_K%s_RS%s_epochs:%s_LR%s' % (args.dataset, args.n_topic, args.seed, epoch+1, args.lr)

                data_dict = {'voc': self.vocabulary,
                             'train_data': 'train',
                             'test_data': test_data,
                             'train_label': train_data['labels'],
                             'test_label': test_data['labels'],
                             'test_llama': self.test_llama}

                evaluation(self, parameter_setting, data_dict, args, self.embedding_model)
                self.model.train()

        if self.model_path is not None:
            self._save_model(self.model_path)

        if self.eval_perplexity and self.model_path is not None:
            self._load_model(self.model_path)
            val_ppl = self._perplexity(train_data)

        return self

    def  eval(self):
        self.model.eval()

    def _save_model(self, model_path):
        assert self.model is not None, \
            'no model to save'

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as file:
            torch.save(self.model, file)

    def _load_model(self, model_path):
        assert os.path.exists(model_path), \
            "model path doesn't exists"

        with open(model_path, 'rb') as file:
            self.model = torch.load(file)
            self.model = self.model.to(self.device)

    def get_theta_train(self) -> torch.Tensor:
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            indices = torch.tensor(range(self.num_docs_train))
            indices = torch.split(indices, self.batch_size)

            thetas = []

            for idx, ind in enumerate(indices):
                data_batch = data.get_batch(
                    self.train_tokens,
                    self.train_counts,
                    ind,
                    self.vocabulary_size,
                    self.device)
                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums if self.bow_norm else data_batch
                theta, _ = self.model.get_theta(normalized_data_batch)

                thetas.append(theta)

            return torch.cat(tuple(thetas), 0)


    def get_theta_test(self) -> torch.Tensor:
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            indices = torch.tensor(range(self.num_docs_test))
            indices = torch.split(indices, self.batch_size)

            thetas = []

            for idx, ind in enumerate(indices):
                data_batch = data.get_batch(
                    self.test_tokens,
                    self.test_counts,
                    ind,
                    self.vocabulary_size,
                    self.device)
                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums if self.bow_norm else data_batch
                theta, _ = self.model.get_theta(normalized_data_batch)

                thetas.append(theta)

            return torch.cat(tuple(thetas), 0)


    def get_doc_top_words(self, topn=10):
        n_test = (len(self.test_llama))
        ind = [i for i in range(n_test)]
        data_batch = data.get_batch(
            self.test_tokens,
            self.test_counts,
            ind,
            self.vocabulary_size,
            self.device)

        sums = data_batch.sum(1).unsqueeze(1)
        if self.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        with torch.no_grad():
            _, _, doc_words = self.model(data_batch, normalized_data_batch)

        sorted_weight, indices = torch.sort(doc_words, dim=1, descending=True)

        top_indx = indices[:,:topn]
        top_weight = sorted_weight[:,:topn]

        top_words = []
        for i in range(top_indx.shape[0]):
            words = itemgetter(*top_indx[i])(self.vocabulary)
            top_words.append(words)

        top_weight = top_weight.cpu().numpy().astype(np.float64)
        top_weight = top_weight/top_weight.sum(axis=1, keepdims=True)

        return top_words, top_weight


