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

We leverage 'llama.cpp' for easy LLM inference, please set it up following [llama.cpp](https://github.com/ggerganov/llama.cpp).
We use llama2-13b-chat in 4-bit quantisation for easy LLM inference, which can be downloaded it [here](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main). Any other LLMs in 'ggml' format that supports llama.cpp also works.

To run LLMs using transformers or get contextualised word embedding from LLMs, please install:
```python
transformers              4.37.2
accelerate                0.26.1
bitsandbytes              0.42.0
```

https://github.com/ggerganov/llama.cpp


https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main


# Dataset

# Get document keywords from LLMs

# Similarity between word sets

# Contextalised word embeddings from LLMs

# Run all evaluation metrics

# Citation
