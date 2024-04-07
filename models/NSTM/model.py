from torch import nn
from sinkhorn import sinkhorn_torch
from torch.nn import functional as F
import torch


class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        self.K = num_class
        super(encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, num_class),
            nn.BatchNorm1d(num_class,eps=0.001, momentum=0.001, affine=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, x, doc_topic, doc_word, M, topic_embedding, sh_alpha, rec_loss_weight):
        sh_loss = sinkhorn_torch(M, doc_topic.t(), doc_word.t(), lambda_sh=sh_alpha).mean()

        rec_log_probs = F.log_softmax(torch.matmul(doc_topic, (1-M)), dim=1)

        rec_loss = -torch.mean(torch.sum((rec_log_probs * x), dim=1))

        joint_loss = rec_loss_weight * rec_loss + sh_loss

        return rec_loss, sh_loss, joint_loss