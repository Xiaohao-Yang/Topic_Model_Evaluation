import os
import re
from tqdm import tqdm


# format prompt for LLAMA2
def llama_v2_prompt(messages):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""None"""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def llm_doc_summary(query_documents, package_path, model_path, embedding_model, topn=30, save_path=None):
    # system prompt with in-context example
    message = [{'role': 'system',
                'content': 'You are an assistant in understanding content of documents. You need to provide %s keywords that capture the essence of the document. The word you provide must be from Glove\'s vocabulary and be single-word. You can use words that are not in the original document. Format your words in a list like this: [word1, word2, word3].' % topn},
               {'role': 'user',
                'content': 'Eastwood High School (El Paso Texas). Eastwood High School is a high school in the city of El Paso Texas USA.'},
               {'role': 'assistant',
                'content': '[eastwood, school, texas, usa, city, education, institution, students, campus, high]'}]

    document_words = []
    for i in tqdm(range(len(query_documents))):
        query_message = [{'role': 'user', 'content': "%s" % query_documents[i]}]
        prompt = message + query_message
        prompt = llama_v2_prompt(prompt) # format prompt

        with open('temp/temp_prompt.txt', 'w') as file:
            file.write(prompt)
            file.close()

        # run llama.cpp inference, --temp 0.01
        command = '%s/./main --model %s -c 2048 --n-predict 512 --repeat_penalty 1.0 --temp 0.01 ' \
                   '--n-gpu-layers 15000 -r \'</s>\' -f temp/temp_prompt.txt > temp/temp_opt.txt' % (package_path, model_path)
        os.system(command)

        # parse output
        with open('temp/temp_opt.txt', 'r') as file:
            text = file.read()
            file.close()
        try:
            # consider possible errors here:
            matches = re.findall(r"\[.*?\]", text)[-1]
            matches = matches.split(',')
            word_list = []
            for item in matches:
                word = item.replace('[', '')
                word = word.replace(']', '')
                word = word.strip().lower()
                word_list.append(word)
        except:
            print('Something wrong with llama output!')
            with open('exception_doc_%s' % save_path, 'a') as file:
                file.write(query_documents[i])
                file.write('\n')
                file.close()
            with open('exception_idx_%s' % save_path, 'a') as file:
                file.write(str(i))
                file.write('\n')
                file.close()

        # filter words to keep only glove words
        glove_words = []
        for word in word_list:
            try:
                embedding = embedding_model[word]
                glove_words.append(word)
            except:
                pass

        if save_path is not None:
            with open(save_path, 'a') as file:
                file.write(' '.join(glove_words))
                file.write('\n')
                file.close()

        document_words.append(glove_words)

    return document_words