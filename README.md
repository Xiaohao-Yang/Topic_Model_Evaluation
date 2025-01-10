# WALM (Word Agreement with Language Model)
This is the official implementation of WALM (**W**ord **A**greement with **L**anguage **M**odel), as proposed in *LLM Reading Tea Leaves: Automatically Evaluating Topic Models with Large Language Models* ([paper link]).

# Installation
```
# clone project
git clone https://github.com/Xiaohao-Yang/Topic_Model_Evaluation.git

# create environment
conda create -n walm python=3.9
conda activate walm

# install pytorch, check for your own system following this page https://pytorch.org/
pip3 install torch torchvision torchaudio

# install required packages
pip install -r requirements.txt
```

# Dataset
We use '20News' and 'DBpedia' (a random subset) for our experiments. The pre-processed datasets can be found in the 'datasets' folder.

We store our pre-processed datasets in '.mat' files, which can be loaded as dictionaries using 'scipy.io.loadmat()'. The datasets/dictionaries have the following common attributes/keys:
* wordsTrain, labelsTrain: bag-of-words (BOW) of training documents, and their labels.
* wordsTest, labelsTest: BOW of testing documents, and their labels.
* vocabulary: vocabularies of the corpus.
* test1, test2: the first and second fold of the test BOWs (for computing document completion perplexity).
* train_text, test_text: text documents of training and testing set.


# Functions
<details>
  <summary>Score Functions</summary>
  
  This is a detailed explanation hidden inside a foldable section. You can add more text here, use Markdown formatting, or even include images or links.
</details>


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
Typically, by extracting keywords from the LLM, we can understand the gist of the document without needing to refer to the original text. Since we only use a quantized model and a single in-context example for inference, there's room for enhancing the summarization accuracy. 

# Similarity between word sets
We propose evaluation metrics for topic models, by quantifying the similarity between keywords from the LLM and the top words from the topic model.
* To get documents' top words from a topic model, check the 'get_doc_word' function of each model's utils.
* To get documents' keywords from the LLM, check 'example.py'.
* To compute the similarity between 2 word sets, check 'word_set_similarity.py'.

# Get contextualised word embeddings from LLMs
For our distance-based evaluation metrics, we consider word embeddings. The word embeddings can be static word embeddings from a pre-trained model such as 'Glove', or a dynamic word embedding that considers the context. Here we provide an example that obtains contextualised word embeddings from an LLM. The functions in the following example can be found in 'llm_embedding.py'.

For this part, we leverage the hugging face 'transformer', with 'bitsandbytes'. Firstly, we set up the model and tokenizer:
```python
llm_paras = {'max_input_length': 2048,
             'base_model': 'meta-llama/Llama-2-13b-chat-hf'
             # 'base_model': 'meta-llama/Llama-2-7b-chat-hf',   # feel free to try different ones
             # 'base_model': 'meta-llama/Llama-2-7b-hf'         
                    }

# quantisation setting
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# load model
model = AutoModelForCausalLM.from_pretrained(llm_paras['base_model'], quantization_config=bnb_config)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    llm_paras['base_model'],
    model_max_length=llm_paras['max_input_length'],
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
```
Here are our example words and context documents for investigation:
```python
# example word and context
word = 'bank'
texts = ["The river bank was flooded",
         "The bank approved my loan application",
         "They save and withdraw money there"]
```
After the setup, we run for the contextualised word embeddings:
```python
embeddings = []
for s in texts:
    # add target words if not mentioned in the document
    if not word in s.strip().split(' '):
        s += ' this document is talking about %s' % word

    # get contextualised embeddings
    word_embedding = get_contextualized_embedding(s, word, model, tokenizer)

    # simply average if multiple target words appear
    if len(word_embedding) > 1:
        word_embedding = torch.stack(word_embedding)
        word_embedding = torch.mean(word_embedding, axis=0)

    embeddings.append(word_embedding[0])
```
Let's check the output contextualised word embeddings:
```python
# the embeddings are different in different contexts
print(embeddings[0])
print(embeddings[1])
print(embeddings[2])

# check similarity
cosine_sim1 = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
cosine_sim2 = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0))
cosine_sim3 = F.cosine_similarity(embeddings[1].unsqueeze(0), embeddings[2].unsqueeze(0))

print(cosine_sim1)
print(cosine_sim2)
print(cosine_sim3)
```

```python
tensor([ 0.6274,  0.8433, -1.8545,  ..., -0.5112,  0.2542,  0.1123],
       dtype=torch.float16)
tensor([ 1.1367,  0.9751, -0.0754,  ..., -0.7827, -0.0156, -0.7827],
       dtype=torch.float16)
tensor([ 0.8745,  1.6484, -1.0928,  ..., -0.7500,  1.0332, -0.1855],
       dtype=torch.float16)

river bank and money bank1 similarity:  tensor([0.6396], dtype=torch.float16)
river bank and money bank2 similarity:  tensor([0.5205], dtype=torch.float16)
money bank1 and money bank2 similarity:  tensor([0.7021], dtype=torch.float16)
```

# Run evaluation for topic models
To run evaluation for topic models:
```python
python main.py --dataset 20News --model NVDM --n_topic 50 --random_seed 1 --epochs 100 --eval_step 10
```
Evaluation output at 'eval_step':
```python
##########################################
Evaluation Phase ...
NVDM_20News_K50_RS1_epochs:10_LR0.001
############################################
doc classification acc:  0.40251358695652173
test ppl:  2991.5
topic diversity:  0.552
purity_test:  0.3179347826086957
nmi_test:  0.2826704840571955
test doc words similarity overlap:  0.025357959581776286
test doc words similarity synset:  0.033568926088255305
test doc words similarity OA:  4.176522543640821
test doc words similarity OT:  0.5356814424001106
##########################################
```
We store the top words for learned topics in a text file for further topic coherence evaluation, which can be done by the [Palmetto](https://github.com/dice-group/Palmetto) package.

# Reference
Our code is based on the following implementations:
* For NVDM: [Code](https://github.com/visionshao/NVDM).
* For PLDA: [Code](https://github.com/estebandito22/PyTorchAVITM).
* For ETM: [Code](https://github.com/lfmatosm/embedded-topic-model).
* For NSTM: [Code](https://github.com/ethanhezhao/NeuralSinkhornTopicModel).
* For SCHOLAR and CLNTM: [Code](https://github.com/nguyentthong/CLNTM).
* For K-means clustering, topic diversity evaluation: [Code](https://github.com/NoviceStone/HyperMiner/tree/main).
* For datasets and document classification evaluation: [Code](https://github.com/Xiaohao-Yang/Topic_Model_Generalisation).


# Citation
Please cite our work if it helps:

