from embedded_topic_model.models.etm import ETM
from utils import load_data_etm, format_data
import argparse
import gensim.downloader as api
import os


parser = argparse.ArgumentParser(description='ETM')
parser.add_argument("--name", type=str, default="ETM")
parser.add_argument("--epochs", help="Number of epoches", default=50, type=int)
parser.add_argument("--dataset", help="Dataset", default='20News')
parser.add_argument("--seed", help="Random seed", default=1, type=int)
parser.add_argument("--n_topic", help="Number of Topics", default=50, type=int)
parser.add_argument("--lr", help="Learning rate", default=0.005, type=float)
parser.add_argument("--eval_step", default=5, type=int)
parser.add_argument("--bs", help="Batch size", default=1000, type=int)
parser.add_argument("--hs", help="Hidden size", default=800, type=int)
args = parser.parse_args()


def main():
    train_data, test_data, word_embeddings, voc, test_llama = load_data_etm(args.dataset)

    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')
    print('Computing ...')

    etm_instance = ETM(
        voc,
        embeddings=word_embeddings,
        num_topics=args.n_topic,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        debug_mode=True,
        batch_size=args.bs,
        t_hidden_size=args.hs,
        test_llama=test_llama,
        embedding_model=model_glove
    )

    etm_instance.fit(train_data, test_data, args)


if __name__ == '__main__':
    etm_data_path = ('models/ETM/datasets/%s' % args.dataset)

    if not os.path.exists(etm_data_path):
        format_data(args.dataset, etm_data_path)

    main()