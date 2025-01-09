import math
from tqdm import tqdm
import re
import random


def create_messages_keywords_suggestion(documents, words_n=10):
    all_messages = []

    for document in documents:
        sys_role = 'You are a helpful assistant in understanding documents.'
        instruction = '''Given a document, understand its content and provide %s indexing words for this document. Note, each of the indexing words is a single word.\nProvide your answer in json format as {'Indexing Words': '<%s Word List>'}.''' % (words_n, words_n)

        messages = []
        messages.append({"role": "system", "content": sys_role + ' ' + instruction})
        messages.append({"role": "user", "content": document})
        all_messages.append(messages)

    return all_messages


def create_messages_word_replacement(word):
    sys_role = 'You are a helpful assistant in understanding words.'
    instruction = '''Provide 10 different words (no compound words allowed) that have similar meaning for the given word. Consider potential noise in the given word. Format your answer in json format as {'Words': '<10 Word List>'}. Do not provide me explanations.'''

    messages = []
    messages.append({"role": "system", "content": sys_role + ' ' + instruction})
    messages.append({"role": "user", "content": word})

    return messages


def create_messages_topic_select(topics, documents):
    all_messages = []
    random.shuffle(topics)
    topic = ', '.join(topics)

    for document in documents:
        sys_role = 'You are a helpful assistant in understanding topics and documents.'
        instruction = '''I have a text corpus talking about the following topics.\n[%s]\n\nGiven a document from the corpus, identify its topics from the provided topic list.\nProvide your answer in json format as {'Topics': '<Selected Topic List>'}.''' % (topic)
        messages = []
        messages.append({"role": "system", "content": sys_role + ' ' + instruction})
        messages.append({"role": "user", "content": document})
        all_messages.append(messages)

    return all_messages


def batch_generator(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def extract_text_between_strings(text, string1, string2):
    pattern = re.escape(string1) + r'(.*?)' + re.escape(string2)
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def generate_keywords(model, tokenizer, docs, save_path, words_n=5, batch_size=1, max_new_tokens=300):
    messages = create_messages_keywords_suggestion(docs, words_n)
    message_batches = batch_generator(messages, batch_size)
    n_batches = math.ceil(len(messages) / batch_size)

    print('Running LLM Inference ...')
    for messages in tqdm(message_batches, total=n_batches):
        encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", truncation=False, padding=True).cuda()
        generated_outputs = model.generate(encodeds,
                                 pad_token_id=tokenizer.pad_token_id,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 num_return_sequences=1,
                                 temperature=None,
                                 top_p=None,
                                 output_scores=True,
                                 return_dict_in_generate=True)

        batch_generated_ids = generated_outputs.sequences.cpu().numpy()
        n_opt = batch_generated_ids.shape[0]
        input_length = encodeds.shape[1]

        for i in range(n_opt):
            # generated text
            generated_sequence = batch_generated_ids[i]
            new_tokens = generated_sequence[input_length:]  # Exclude the input tokens
            decoded_sequence = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # parse answer
            answer = decoded_sequence.replace('"', "'")
            try:
                words = extract_text_between_strings(answer, "'Indexing Words': [", ']')[0].strip()
                words = words.replace("'", '').split(',')
                words = [w.strip() for w in words]
                ans_dict = {'Words': words}
            except:
                print('Error when parsing answer:')
                print(answer)
                ans_dict = None

            # save answer to file
            with open(save_path, 'a') as file:
                file.write(str(ans_dict) + '\n')


def generate_word_replacement(model, tokenizer, word, max_new_tokens=100):
    messages = create_messages_word_replacement(word)
    encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                             truncation=False, padding=True).cuda()
    generated_outputs = model.generate(encodeds,
                                       pad_token_id=tokenizer.pad_token_id,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=False,
                                       num_return_sequences=1,
                                       temperature=None,
                                       top_p=None,
                                       output_scores=True,
                                       return_dict_in_generate=True)

    batch_generated_ids = generated_outputs.sequences.cpu().numpy()
    input_length = encodeds.shape[1]

    # generated text
    generated_sequence = batch_generated_ids[0]
    new_tokens = generated_sequence[input_length:]  # Exclude the input tokens
    decoded_sequence = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # parse answer
    decoded_sequence = decoded_sequence.replace('"', "'")

    try:
        words = extract_text_between_strings(decoded_sequence, "Words': [", "]")[0]
    except:
        print('Error when parsing ... ')
        print(decoded_sequence)
        words = 'None'
    words = words.replace("'", '').split(',')
    words = [w.strip() for w in words]
    return words


def generate_topic_select(model, tokenizer, topics, documents, save_path, batch_size=1, max_new_tokens=300):
    messages = create_messages_topic_select(topics, documents)
    message_batches = batch_generator(messages, batch_size)
    n_batches = math.ceil(len(messages) / batch_size)

    print('Running LLM Inference ...')
    for messages in tqdm(message_batches, total=n_batches):
        encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", truncation=False, padding=True).cuda()
        generated_outputs = model.generate(encodeds,
                                 pad_token_id=tokenizer.pad_token_id,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 num_return_sequences=1,
                                 temperature=None,
                                 top_p=None,
                                 output_scores=True,
                                 return_dict_in_generate=True)

        batch_generated_ids = generated_outputs.sequences.cpu().numpy()
        n_opt = batch_generated_ids.shape[0]
        input_length = encodeds.shape[1]

        for i in range(n_opt):
            # generated text
            generated_sequence = batch_generated_ids[i]
            new_tokens = generated_sequence[input_length:]
            decoded_sequence = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # parse answer
            answer = decoded_sequence.replace('"', "'")
            try:
                topics = extract_text_between_strings(answer, "'Topics': [", ']')[0].strip()
                topics = topics.replace("'", '').split(',')
                topics = [w.strip() for w in topics]
                ans_dict = {'Topics': topics}
            except:
                print('Error when parsing answer:')
                print(answer)
                ans_dict = None

            with open(save_path, 'a') as file:
                file.write(str(ans_dict) + '\n')


def generate_topics_aware_kw(model, tokenizer, topics, documents, save_path, words_n=5, max_new_tokens=300):
    for i in tqdm(range(len(topics))):
        if not topics[i].strip() == 'None':
            ts = eval(topics[i])['Topics']

            keywords = []
            for topic in ts:
                # create message
                sys_role = 'You are a helpful assistant in understanding topics and documents.'
                instruction = '''I have a document talking about topic -- "%s".\nBased on this topic, suggest %s indexing words for this document.\nProvide your answer in json format as {'Indexing Words': '<%s Word List>'}.''' % (topic, words_n, words_n)
                messages = []
                messages.append({"role": "system", "content": sys_role + ' ' + instruction})
                messages.append({"role": "user", "content": documents[i]})

                # inference
                encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                         truncation=False, padding=True).cuda()
                generated_outputs = model.generate(encodeds,
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   max_new_tokens=max_new_tokens,
                                                   do_sample=False,
                                                   num_return_sequences=1,
                                                   temperature=None,
                                                   top_p=None,
                                                   output_scores=True,
                                                   return_dict_in_generate=True)

                batch_generated_ids = generated_outputs.sequences.cpu().numpy()
                input_length = encodeds.shape[1]

                # generated text
                generated_sequence = batch_generated_ids[0]
                new_tokens = generated_sequence[input_length:]
                decoded_sequence = tokenizer.decode(new_tokens, skip_special_tokens=True)

                # parse answer
                answer = decoded_sequence.replace('"', "'")
                try:
                    words = extract_text_between_strings(answer, "'Indexing Words': ", '}')[0].strip()
                    words = words.replace("[", '').replace(']', '')
                    words = words.replace("'", '').split(',')
                    words = [w.strip() for w in words]
                except:
                    print('--------')
                    print('Error when parsing answer:')
                    print(answer)
                    print('--------')
                    words = []
                keywords += words
            keywords = list(set(keywords))
        else:
            keywords = []

        with open(save_path, 'a') as file:
            if len(keywords) > 0:
                file.write(', '.join(keywords) + '\n')
            else:
                 file.write('None' + '\n')


def replace_oov(list_of_words, model, tokenizer, embedding_model):
    filtered_words = []
    for i in range(len(list_of_words)):
        new_words = []
        if not list_of_words[i].strip() == 'None':
            llm_words = list_of_words[i].split(',')
            llm_words = [w.strip().lower() for w in llm_words]

            # repalce oov
            for word in llm_words:
                replace = None
                try:
                    embedding_model[word]
                    new_words.append(word)
                except:
                    replace = True

                if replace:
                    rep_words = generate_word_replacement(model, tokenizer, word)
                    if not rep_words == 'None':
                        found = False
                        for rep in rep_words:
                            try:
                                embedding_model[rep]
                                found = True
                                new_words.append(rep)
                                break
                            except:
                                pass
                        if not found:
                            print('Fail to find replacement for word: %s' % word)
        else:
            new_words.append('None')

        filtered_words.append(new_words)

    return filtered_words
