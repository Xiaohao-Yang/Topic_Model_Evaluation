# Topic Model Evaluation
Evaluation of topic model generative quality with large language models.

# Requirements
```python
torch: 2.1.2 
torchaudio: 2.1.2                    
torchvision: 0.16.2 
scikit-learn: 1.4.0
numpy: 1.26.3 
nltk: 3.8.1
scipy: 1.12.0
spacy: 3.7.4
gensim: 3.8.3         
tqdm: 4.66.1
pot: 0.9.3
```

We leverage 'llama.cpp' for easy LLM inference, please set it up following [llama.cpp](https://github.com/ggerganov/llama.cpp) (We suggest BLAS Build with CUDA for faster inference.); The LLM we use is 'llama2-13b-chat' in 4-bit quantisation, which can be downloaded [here](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main). We are not specific to a certainty LLM, any other LLMs in 'ggml' format that supports 'llama.cpp' also work.

To run LLMs using 'hugging face' transformers or get contextualised word embeddings from LLMs, please install:
```python
transformers: 4.37.2
accelerate: 0.26.1
bitsandbytes: 0.42.0
```

# Dataset
We use '20News' and 'DBpedia' (a random subset) for our experiments. The pre-processed datasets can be found in the 'datasets' folder.

We store our pre-processed datasets in '.mat' files, which can be loaded as dictionaries using 'scipy.io.loadmat()'. The datasets/dictionaries have the following common attributes/keys:
* train_data, train_label: bag-of-words (BOW) of training documents, and their labels.
* test_data, test_label: BOW of testing documents, and their labels.
* voc, word_embeddings: vocabularies of the corpus, and their word embeddings from 'glove-wiki-gigaword-50'.
* test1, test2: the first and second fold of the test BOWs (for computing document completion perplexity).
* train_text, test_text: text documents of training and testing set.
* train_llama, test_llama: document keywords obtained from the LLM of training and testing documents. We run the LLM for keyword summarisation once for all the documents and store it here, to reduce resources.


# Get document keywords from LLMs
Here, we provide an example of document keyword summarisation by an LLM, which is also illustrated in 'example.py'.

Firstly, we randomly pick some documents from '20News' and 'DBpedia':
```python
# 20News
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')).data
newsgroups = [process_huamn_readable(text) for text in newsgroups]  # or define your own 'process_huamn_readable'
newsgroups = newsgroups[1000:1010]

# DBpedia
with open('datasets/DBpedia.txt', 'r') as file:
    DBpedia = file.readlines()
    file.close()
DBpedia = DBpedia[1000:1010]
```
Then, we define 1) the path to the LLM model you want to use, and 2) the path to the 'llama.cpp' package, and 3) the number of keywords you expected:
```python
# llm model path
# download the model from hugging face and define your own model path
model_path = '~/Desktop/llama.cpp/LLMs/llama-2-13b-chat.ggmlv3.q4_1.bin'

# configure llama.cpp and define the package path
package_path = '~/Desktop/llama.cpp'

# expected keywords
topn = 30
```
As the words obtained from LLMs are from a huge space, we further filter the words obtained from the LLM and keep only the words in GloVe's vocabulary set, which makes it cheap to get their word embeddings.
```python
# model for word embeddings
embedding_model = 'glove-wiki-gigaword-50'
print('Loading embedding model %s...' % embedding_model)
embedding_model = api.load("glove-wiki-gigaword-50")
print('Loading done!')
```
Following that, we set up the path to save the output, and run the summarisation function:
```python
save_path_news = 'test_20News.txt'     # or define your save path
save_path_DBpedia = 'test_DBpedia.txt' # or define your save path

doc_words_news = llm_doc_summary(newsgroups, package_path, model_path, embedding_model, topn, save_path_news)
doc_words_DBpedia = llm_doc_summary(DBpedia, package_path, model_path, embedding_model, topn, save_path_DBpedia)
```

Here are some output keywords summarisation by the LLM, for the original documents:
```python
# origianl documents
Dojutr w. Dojutr w d jutruf is a village in the administrative district of Gmina Blizan w within Kalisz County Greater Poland Voivodeship in west central Poland.
# keywords from the LLM
village poland district gmina kalisz county voivodeship west central

# origianl documents
Tony Souli . Tony Souli (born 1955) is a French artist working in painting printmaking sculpture installation art and photography.
# keywords from the LLM
artist french painting printmaking sculpture installation photography

# origianl documents
I 121 class submarine. The I 121 class submarine ( I 121 gata Sensuikan) was a class of submarine in the Imperial Japanese Navy (IJN) serving from the 1920s to the Second World War. The IJN classed it as a Kiraisen ( Minelaying submarine).
# keywords from the LLM
submarine japanese navy world war minelaying class imperial service
```
Typically, by extracting keywords from the LLM, we can understand the gist of the document without needing to refer to the original text. Since we only use a small, quantized model and a single in-context example for inference, there's room for enhancing the summarization accuracy. 

# Similarity between word sets

# Get contextalised word embeddings from LLMs

# Run all evaluation metrics

# Reference

# Citation
