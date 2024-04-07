'''
run llama2 keywords summarisation for documents from 20News and DBpedia
'''
import re
from sklearn.datasets import fetch_20newsgroups
import gensim.downloader as api
from llm_keywords import llm_doc_summary


# for 20news dataset
def process_huamn_readable(text):
    cleaned_string = re.sub(r'\n\n', '', text)
    cleaned_string = re.sub(r'\n', ' ', cleaned_string)
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)

    return cleaned_string


if __name__ == '__main__':
    # 20News
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')).data
    newsgroups = [process_huamn_readable(text) for text in newsgroups]

    # DBpedia
    with open('datasets/DBpedia.txt', 'r') as file:
        DBpedia = file.readlines()
        file.close()

    # llama2 model path
    # download the model from huggingface and define your own model path
    model_path = '~/Desktop/llama.cpp/LLMs/llama-2-13b-chat.ggmlv3.q4_1.bin'
    # configure llama.cpp and define the package path
    package_path = '~/Desktop/llama.cpp'
    # expected keywords
    topn = 30

    # model for word embeddings
    embedding_model = 'glove-wiki-gigaword-50'
    print('Loading embedding model %s...' % embedding_model)
    embedding_model = api.load("glove-wiki-gigaword-50")
    print('Loading done!')
    print('Computing ...')

    # run for 20News
    newsgroups = newsgroups[1000:1010]
    save_path_news = 'test_20News.txt' # or define your save path
    doc_words_news = llm_doc_summary(newsgroups, package_path, model_path, embedding_model, topn, save_path_news)

    # run for DBpedia
    DBpedia = DBpedia[1000:1010]
    save_path_DBpedia = 'test_DBpedia.txt' # or define your save path
    doc_words_DBpedia = llm_doc_summary(DBpedia, package_path, model_path, embedding_model, topn, save_path_DBpedia)

    for item in doc_words_news:
        print(' '.join(item))
    for item in doc_words_DBpedia:
        print(' '.join(item))

    '''
    The empty line in the output file means all words for this document are filtered out 
    because they are not in glove's vocabulary. 
    '''