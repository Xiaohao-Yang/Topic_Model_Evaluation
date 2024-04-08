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

We leverage 'llama.cpp' for easy LLM inference, please set it up following [llama.cpp](https://github.com/ggerganov/llama.cpp); The LLM we use is 'llama2-13b-chat' in 4-bit quantisation, which can be downloaded [here](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main). We are not specific to a certainty LLM, any other LLMs in 'ggml' format that supports 'llama.cpp' also work.

To run LLMs using 'hugging face' transformers or get contextualised word embedding from LLMs, please install:
```python
transformers: 4.37.2
accelerate: 0.26.1
bitsandbytes: 0.42.0
```

# Dataset

# Get document keywords from LLMs

# Similarity between word sets

# Contextalised word embeddings from LLMs

# Run all evaluation metrics

# Reference

# Citation
