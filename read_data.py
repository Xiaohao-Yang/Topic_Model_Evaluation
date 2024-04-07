import scipy.io as sio
from scipy import sparse


def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix


def load_data(mat_file_name, is_to_dense=True):
    data = sio.loadmat(mat_file_name)
    train_data = data['wordsTrain'].transpose()
    test_data = data['wordsTest'].transpose()

    try:
        word_embeddings = data['embeddings']
    except:
        word_embeddings = None

    voc = data['vocabulary']
    voc = [v[0][0] for v in voc]
    test1 = data['test1'].transpose()
    test2 = data['test2'].transpose()
    train_label = data['labelsTrain']
    test_label = data['labelsTest']

    if is_to_dense:
        train_data = sparse2dense(train_data)
        test_data = sparse2dense(test_data)
        test1 = sparse2dense(test1)
        test2 = sparse2dense(test2)

    # load text data
    try:
        train_text, test_text = data['train_text'], data['test_text']
        train_text = [item[0][0].strip() for item in train_text]
        test_text = [item[0][0].strip() for item in test_text]
    except:
        train_text, test_text = None, None

    # load llama output
    try:
        train_llama, test_llama = data['train_llama'], data['test_llama']
        train_llama = [item[0][0].strip() for item in train_llama]
        test_llama = [item[0][0].strip() for item in test_llama]
    except:
        train_llama, test_llama = None, None

    data_dict = {
        'train_data': train_data,
        'train_label': train_label,
        'test_data': test_data,
        'test_label': test_label,
        'word_embeddings': word_embeddings,
        'voc': voc,
        'test1': test1,
        'test2': test2,
        'train_text': train_text,
        'test_text': test_text,
        'train_llama': train_llama,
        'test_llama' :test_llama
    }

    return data_dict
