'''
obtain contextualised embeddings from huggingface llm
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F


def get_contextualized_embedding(doc, word, model, tokenizer):
    inputs = tokenizer(doc, return_tensors="pt")
    input_ids = inputs.input_ids.numpy()[0]
    word_token = tokenizer(word, return_tensors="pt").input_ids.numpy()[0]
    word_positions = [i for i, token in enumerate(input_ids) if token == word_token[1]]

    if not word_positions:
        print(f"The word '{word}' is not found in the provided sentence.")
        return []

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    word_embeddings = []
    for pos in word_positions:
        word_embedding = hidden_states[-1][0, pos]
        word_embeddings.append(word_embedding)

    return word_embeddings


if __name__ == '__main__':
    llm_paras = {'max_input_length': 2048,
                    'base_model': 'meta-llama/Llama-2-13b-chat-hf'
                    # 'base_model': 'meta-llama/Llama-2-7b-chat-hf',
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

    # example word and context
    word = 'bank'
    texts = ["The river bank was flooded",
             "The bank approved my loan application",
             "They save and withdraw money there"]

    embeddings = []
    for s in texts:
        # add target words if not mentioned in the document
        if not word in s.strip().split(' '):
            s += ' this document is talking about %s' % word

        # get contextualised embeddings
        word_embedding = get_contextualized_embedding(s, word, model, tokenizer)

        # average if multiple target words appear
        if len(word_embedding) > 1:
            word_embedding = torch.stack(word_embedding)
            word_embedding = torch.mean(word_embedding, axis=0)

        embeddings.append(word_embedding[0])

    # the embeddings are different in different context
    print(embeddings[0])
    print(embeddings[1])
    print(embeddings[2])

    cosine_sim1 = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    cosine_sim2 = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0))
    cosine_sim3 = F.cosine_similarity(embeddings[1].unsqueeze(0), embeddings[2].unsqueeze(0))

    print(cosine_sim1)
    print(cosine_sim2)
    print(cosine_sim3)