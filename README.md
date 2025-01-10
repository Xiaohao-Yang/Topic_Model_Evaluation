# WALM (Word Agreement with Language Model)
This is the official implementation of WALM (Word Agreement with Language Model), as proposed in *LLM Reading Tea Leaves: Automatically Evaluating Topic Models with Large Language Models* ([paper link]).

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
  
  ## WALM score functions

WALM score functions compute the similarity between two sets of words.


```python
from walm import walm_overlap, walm_synset, walm_ot, walm_oa
import gensim.downloader as api # we use gensim word embedding models
```

    [nltk_data] Downloading package wordnet to /home/xiaohao/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!



```python
words1 = ['us', 'au']
words2 = ['america', 'australia', 'people']
```

### Overlap-based score functions


```python
# overlap based scores
print('walm overlap: ', walm_overlap(words1, words2))
print('walm synset: ', walm_synset(words1, words2))
```

    walm overlap:  0.0
    walm synset:  0.2


walm_overlap measures the overlap between two sets of words, while walm_synset extends this by considering synset overlap between different words.

### Embedding-based score functions


```python
# load word embedding model
print('Loading glove model ...')
embedding_model = api.load("glove-wiki-gigaword-50")
print('Loading done!')
```

    Loading glove model ...
    Loading done!



```python
# optimal assignment
print('walm optimal assignment: ', walm_oa(words1, words2, embedding_model))
```

    walm optimal assignment:  0.978635346639759


walm_oa solves an optimal assignment problem between word set 1 and word set 2.


```python
# optimal transport
word_dis1 = {'words': words1, 'weights': [0.5, 0.5]}
word_dis2 = {'words': words2, 'weights': [0.1, 0.1, 0.8]}
print('walm optimal transport: ', walm_ot(word_dis1, word_dis2, embedding_model))
```

    walm optimal transport:  0.5193005172391889


walm_ot solves an optimal transport problem between word distribution 1 and word distribution 2.
</details>


<details>
  <summary>Keywords from LLMs</summary>
  
  This is a detailed explanation hidden inside a foldable section. You can add more text here, use Markdown formatting, or even include images or links.
</details>


<details>
  <summary>Topic-Aware Keywords from LLMs</summary>
  
  This is a detailed explanation hidden inside a foldable section. You can add more text here, use Markdown formatting, or even include images or links.
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

