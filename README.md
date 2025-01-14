# Topic model evaluation with WALM (Word Agreement with Language Model)
This is the official implementation of WALM (Word Agreement with Language Model), as proposed in: *LLM Reading Tea Leaves: Automatically Evaluating Topic Models with Large Language Models* ([paper link](https://arxiv.org/abs/2406.09008)).

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
* vocabulary: vocabulary set of the text corpus.
* test1, test2: the first and second fold of the test BOWs (for computing document completion perplexity).
* train_text, test_text: text documents of training and testing set.


# Functions
<details>
  <summary>Word Set Similarity</summary>
  
  ## WALM score functions ([jupyter notebook](score_functions.ipynb))

WALM score functions compute the similarity between two sets of words.


```python
from walm import walm_overlap, walm_synset, walm_ot, walm_oa
import gensim.downloader as api # we use gensim word embedding models
```





```python
words1 = ['us', 'au']
words2 = ['america', 'australia', 'people']
```

### Overlap-based score functions
walm_overlap measures the overlap between two sets of words, while walm_synset extends this by considering synset overlap between different words.

```python
# overlap based scores
print('walm overlap: ', walm_overlap(words1, words2))
print('walm synset: ', walm_synset(words1, words2))
```

    walm overlap:  0.0
    walm synset:  0.2


### Embedding-based score functions
walm_oa solves an optimal assignment problem between word set 1 and word set 2. walm_ot solves an optimal transport problem between word distribution 1 and word distribution 2.

```python
# load word embedding model
print('Loading glove model ...')
embedding_model = api.load("glove-wiki-gigaword-50")
print('Loading done!')
```



```python
# optimal assignment
print('walm optimal assignment: ', walm_oa(words1, words2, embedding_model))
```

    walm optimal assignment:  0.978635346639759





```python
# optimal transport
word_dis1 = {'words': words1, 'weights': [0.5, 0.5]}
word_dis2 = {'words': words2, 'weights': [0.1, 0.1, 0.8]}
print('walm optimal transport: ', walm_ot(word_dis1, word_dis2, embedding_model))
```

    walm optimal transport:  0.5193005172391889


</details>


<details>
  <summary>Generate Keywords from LLMs</summary>
  
  ## Generate keywords for test documents from an LLM ([jupyter notebook](kw_llm.ipynb))


```python
from walm import generate_keywords
from transformers import AutoModelForCausalLM, AutoTokenizer
import scipy.io as sio
import torch
```




```python
# load documents
dataset = '20News'
data_dict = sio.loadmat('datasets/%s/data.mat' % dataset)
test_doc = data_dict['test_text'].tolist()
test_doc = [doc[0][0].strip() for doc in test_doc]

# take 10 documents as an example
test_doc = test_doc[0:10]
```


```python
# load an llm model, we support the following LLMs in current version
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
# model_name = 'microsoft/Phi-3-mini-128k-instruct'
# model_name = '01-ai/Yi-1.5-9B-Chat'

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16
                                             ).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```





```python
# generate keywords
save_path = 'test_opt.txt'
outputs = generate_keywords(model, tokenizer, test_doc, save_path)
```


The output will be also saved in your defined .txt file in JSON format:


```python
{'Words': ['Harvard', 'Graphics', 'Windows', 'Sale', 'Price']}
{'Words': ['Abortion', 'Rights', 'Case', 'Good', 'Pro']}
{'Words': ['mound', 'season', 'strike', 'zone', 'seventies']}
{'Words': ['PlusMinus', 'Players', 'Role', 'Time', 'Jagr']}
{'Words': ['Efficiency', 'X11', 'Clients', 'Performance', 'XRemote']}
{'Words': ['Windows', 'Hardware', 'Interrupts', 'DOS', 'UART']}
{'Words': ['train', 'station', 'cities', 'drive', 'airport']}
{'Words': ['IDE', 'Drive', 'Error', 'Format', 'Software']}
{'Words': ['Projector', 'Super', '8mm', 'Sound', 'ForSale']}
{'Words': ['Society', 'Life', 'Killing', 'Value', 'Children']}
```

  
</details>


<details>
  <summary>Generate Topic-Aware Keywords from LLMs</summary>
  
  ## Generate topic-aware keywords for test documents from LLM ([jupyter notebook](kw-topic_llm.ipynb))

### Step1: Generate global topics

To generate global topics for the document collection from an LLM, we follow the topic generation approach in [TopicGPT](https://github.com/chtmp223/topicGPT). We have saved the output topics under the dataset folder (e.g., datasets/20News/topics.txt).

### Step2: Topic selection


```python
from walm import extract_text_between_strings, generate_topic_select, generate_topics_aware
from transformers import AutoModelForCausalLM, AutoTokenizer
import scipy.io as sio
import torch

# load documents
dataset = '20News'
data_dict = sio.loadmat('datasets/%s/data.mat' % dataset)
test_doc = data_dict['test_text'].tolist()
test_doc = [doc[0][0].strip() for doc in test_doc]

# take 10 documents as an example
test_doc = test_doc[0:10]

# load llm model
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
# model_name = 'microsoft/Phi-3-mini-128k-instruct'
# model_name = '01-ai/Yi-1.5-9B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16
                                             ).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# load global topics from this dataset
with open('datasets/%s/topics.txt' % dataset, 'r') as file:
    topics = file.readlines()
topics = topics[0:-1]
topics = [extract_text_between_strings(item, "[1]", "(Count:")[0].strip() for item in topics]

# run topic selection
save_path = 'step1_topic-select.txt'
doc_topics = generate_topic_select(model, tokenizer, topics, test_doc, save_path)
```



```python
for item in doc_topics:
    print(item)
```

    {'Topics': ['Technology and Electronics', 'Software Development', 'Economic and Employment Issues']}
    {'Topics': ['Consumer Rights', 'Law and Justice', 'Ethics', 'Society']}
    {'Topics': ['Sports Statistics', 'History']}
    {'Topics': ['Sports Statistics']}
    {'Topics': ['Technology and Electronics', 'Software Development', 'Internet Culture', 'Communication']}
    {'Topics': ['Technology and Electronics', 'Software Development']}
    {'Topics': ['Transportation', 'Consumer Rights', 'Society']}
    {'Topics': ['Technology and Electronics', 'Repair and Maintenance', 'Software Development']}
    {'Topics': ['Technology and Electronics', 'Consumer Rights']}
    {'Topics': ['Society', 'Ethics', 'Religion']}


### Step2: Topic-Aware keywords generation


```python
# load selected topics from test documents
with open('step1_topic-select.txt', 'r') as file:
    doc_topics = file.readlines()

# define save path and run generation
save_path = 'step2_topic-aware_kws.txt'
outputs = generate_topics_aware(model, tokenizer, doc_topics, test_doc, save_path)
```

     



```python
for item in outputs:
    print(item)
```

    Technology, Development, Sale, Software, Economic, Graphics, Harvard, Employment, Electronics, Issues, Price, Windows, Harvard Graphics
    Ethical, Justice, Constitution, Societal, Morality, Abortion, Bioethics, Moral, Rights, Ethics, Law
    History, Strike, Sports, Baseball, Mound, Zone, Statistics
    Role, Plus/minus, Players, Context, Statistics
    Async Solutions, Graphics Accelerator, Efficiency, Internet Culture, X11 Clients, Communication, Software Development, X11, Clients
    Technology, UART, Serial Communication, Software Development, Electronics, Hardware Interrupts, Interrupts, Hardware, Windows
    Train, Station, City, Protection, Dispute, Airport, Cities, Infrastructure, Complaint, Consumer, Rights, Society, Transportation
    PCTools, Data Error, Error, IDE drive, IDE, Drive, Low Level Format, Maintenance, Data error, Disk recovery, IDE Drive, Repair, Low level format, Sector Marking, Disk Recovery
    Technology, Sound, Sale, Protection, Purchase, Electronics, Super 8mm, Projector, Consumer, Rights
    Life, Morality, Violence, Killing, Moral, Values, Society, Value

</details>


<details>
  <summary>OOV handling by the LLM</summary>

  # Repalce OOV (Out-Of-Vocabulary) words by the LLM ([jupyter notebook](replace_oov.ipynb))

The keywords of documents generated by an LLM may contain out-of-vocabulary (OOV) words. Here, we illustrate how OOV words can be replaced using an LLM.


```python
from walm import replace_oov
import gensim.downloader as api
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
```



```python
# load keyword file
kw_file = 'step2_topic-aware_kws.txt'
with open(kw_file, 'r') as file:
    words_list = file.readlines()

print('Original keywords for documents:')
for i in range(len(words_list)):
    print('document %s: ' % i + words_list[i].strip())
```

    Original keywords for documents:
    document 0: Technology, Development, Sale, Software, Economic, Graphics, Harvard, Employment, Electronics, Issues, Price, Windows, Harvard Graphics
    document 1: Ethical, Justice, Constitution, Societal, Morality, Abortion, Bioethics, Moral, Rights, Ethics, Law
    document 2: History, Strike, Sports, Baseball, Mound, Zone, Statistics
    document 3: Role, Plus/minus, Players, Context, Statistics
    document 4: Async Solutions, Graphics Accelerator, Efficiency, Internet Culture, X11 Clients, Communication, Software Development, X11, Clients
    document 5: Technology, UART, Serial Communication, Software Development, Electronics, Hardware Interrupts, Interrupts, Hardware, Windows
    document 6: Train, Station, City, Protection, Dispute, Airport, Cities, Infrastructure, Complaint, Consumer, Rights, Society, Transportation
    document 7: PCTools, Data Error, Error, IDE drive, IDE, Drive, Low Level Format, Maintenance, Data error, Disk recovery, IDE Drive, Repair, Low level format, Sector Marking, Disk Recovery
    document 8: Technology, Sound, Sale, Protection, Purchase, Electronics, Super 8mm, Projector, Consumer, Rights
    document 9: Life, Morality, Violence, Killing, Moral, Values, Society, Value



```python
# We use the vocabulary set of GloVe model, we load the Gensim GloVe model firstly.
print('Loading glove model ...')
embedding_model = api.load("glove-wiki-gigaword-50")
print('Loading done!')
```

```python
# LLM for word repalce for OOV
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
llm = AutoModelForCausalLM.from_pretrained(model_name,
                                           trust_remote_code=True,
                                           torch_dtype=torch.float16
                                           ).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
filtered_words = replace_oov(words_list, llm, tokenizer, embedding_model)
```


```python
print('Keywords for documents after filtering:')
for i in range(len(filtered_words)):
    print('document %s: ' % i + ', '.join(filtered_words[i]))
```

    Keywords for documents after filtering:
    document 0: technology, development, sale, software, economic, graphics, harvard, employment, electronics, issues, price, windows, art
    document 1: ethical, justice, constitution, societal, morality, abortion, bioethics, moral, rights, ethics, law
    document 2: history, strike, sports, baseball, mound, zone, statistics
    document 3: role, plus/minus, players, context, statistics
    document 4: instant, efficiency, cyberspace, communication, programming, x11, clients
    document 5: technology, uart, programming, electronics, interrupts, interrupts, hardware, windows
    document 6: train, station, city, protection, dispute, airport, cities, infrastructure, complaint, consumer, rights, society, transportation
    document 7: tools, mistake, error, mind, ide, drive, base, maintenance, mistake, retrieval, mind, repair, base, mark, retrieval
    document 8: technology, sound, sale, protection, purchase, electronics, small, projector, consumer, rights
    document 9: life, morality, violence, killing, moral, values, society, value
  
</details>


<details>
  <summary>Keywords from Topic Models</summary>
  
  This is a detailed explanation hidden inside a foldable section. You can add more text here, use Markdown formatting, or even include images or links.
</details>

<details>
  <summary>Contextualized word embeddings from LLMs</summary>
  
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
</details>






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


# Citation
Please cite our work if it helps:
```python
@article{yang2024llm,
  title={LLM Reading Tea Leaves: Automatically Evaluating Topic Models with Large Language Models},
  author={Yang, Xiaohao and Zhao, He and Phung, Dinh and Buntine, Wray and Du, Lan},
  journal={arXiv preprint arXiv:2406.09008},
  year={2024}
}
```
