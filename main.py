import argparse
import os
from parameters import parameter_dict


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20News')
parser.add_argument('--model', type=str, default='NVDM')
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--eval_step', type=int, default=5)
args = parser.parse_args()


if __name__ == '__main__':
    setting = str(args.dataset) + '_K' + str(args.n_topic)   # dataset and K

    if args.model == 'LDA':
        argument = ('python models/%s/train.py --dataset=%s --n_topic=%s --seed=%s --epochs=%s --eval_step=%s' %
                    (args.model, args.dataset, args.n_topic, args.random_seed, args.epochs, args.eval_step))

    elif args.model == 'NSTM':
        paras = parameter_dict[args.model][args.dataset]
        lr = paras[0]           # learning rate
        rec_weight = paras[1]   # reconstruction weight
        epochs = paras[2]       # training epochs
        argument = (('python models/%s/train.py --dataset=%s --n_topic=%s --seed=%s --epochs=%s --eval_step=%s '
                    '--lr=%s --rec_loss_weight=%s') %
                    (args.model, args.dataset, args.n_topic, args.random_seed, epochs, args.eval_step, lr, rec_weight))

    elif args.model in ['SCHOLAR', 'CLNTM']:
        paras = parameter_dict[args.model][setting]
        lr = paras[0]           # learning rate
        epochs = paras[1]       # training epochs
        argument = ('python models/SCHOLAR/train.py --model %s --dataset=%s --n_topic=%s --seed=%s --epochs=%s --eval_step=%s --lr=%s') \
                   % (args.model, args.dataset, args.n_topic, args.random_seed, epochs, args.eval_step, lr)
    else:
        paras = parameter_dict[args.model][setting]
        lr = paras[0]                                            # learning rate
        epochs = paras[1]                                        # training epochs
        argument = ('python models/%s/train.py --dataset=%s --n_topic=%s --seed=%s --epochs=%s --eval_step=%s --lr=%s') \
                   % (args.model, args.dataset, args.n_topic, args.random_seed, epochs, args.eval_step, lr)

    os.system(argument)