import torch
import torch.nn as nn
import torch.nn.functional as F


class NVDM(nn.Module):
    def __init__(self, vocab_size, n_hidden, n_topic, n_sample):
        super(NVDM, self).__init__()

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample

        # encoder architecture
        # encode doc to vectors
        self.enc_vec = nn.Linear(self.vocab_size, self.n_hidden)
        # get mean of Gaussian distribution
        self.mean = nn.Linear(self.n_hidden, self.n_topic)
        # get log_sigma of Gaussian distribution
        self.log_sigma = nn.Linear(self.n_hidden, self.n_topic)

        # decoder architecture
        self.dec_vec = nn.Linear(self.n_topic, self.vocab_size)

    def encoder(self, x):
        # encode doc to vectors
        enc_vec = torch.tanh(self.enc_vec(x))
        # getting variational parameters
        mean = self.mean(enc_vec)
        log_sigma = self.log_sigma(enc_vec)
        # computing kld
        kld = -0.5 * torch.sum(1 - torch.square(mean) + 2 * log_sigma - torch.exp(2 * log_sigma), 1)
        return mean, log_sigma, kld

    def decoder(self, mean, log_sigma, x):
        # reconstruct doc from encoded vector
        if self.n_sample == 1:  # single sample
            eps = torch.rand(self.batch_size, self.n_topic).cuda()
            doc_vec = torch.mul(torch.exp(log_sigma), eps) + mean
            logits = F.log_softmax(self.dec_vec(doc_vec), dim=1)
            recons_loss = -torch.sum(torch.mul(logits, x), 1)
        # multiple samples
        else:
            eps = torch.rand(self.n_sample * self.batch_size, self.n_topic)
            eps_list = list(eps.view(self.n_sample, self.batch_size, self.n_topic))
            recons_loss_list = []
            for i in range(self.n_sample):
                curr_eps = eps_list[i]
                doc_vec = torch.mul(torch.exp(log_sigma), curr_eps) + mean
                logits = F.log_softmax(self.dec_vec(doc_vec))
                recons_loss_list.append(-torch.sum(torch.mul(logits, x), 1))
            recons_loss_list = torch.tensor(recons_loss_list)
            recons_loss = torch.sum(recons_loss_list, dim=1) / self.n_sample

        return recons_loss, doc_vec, logits

    def forward(self, x):
        self.batch_size = len(x)
        mean, log_sigma, kld = self.encoder(x)
        recons_loss, doc_vec, doc_word = self.decoder(mean, log_sigma, x)

        return kld, recons_loss, mean, doc_vec, doc_word