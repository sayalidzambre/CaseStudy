import numpy as np
import pandas as pd

def get_doc_matrix(w2v_model, corpora_token):
    """
    Function to aggregate document vector from a built gensim w2v model that
    calculates the vector mean based on vector representation of words in the
    document
    -----PARAMETERS-----
    w2v_model: a gensim.models.Word2Vec object that can return a numeric array
        when queried with w2v_model['word']
    corpora_token: a list of sentence in list form, e.g. [['sentence','one'],
        ['sentence','two'],...]
    -----OUTPUT-----
    returned object (text_matrix) is a numpy.ndarray with the shape
    (len(corpora_token), word_vector_length)
    """
    word_vector_length = len(w2v_model[w2v_model.wv.index2word[0]])
    text_matrix = np.zeros((len(corpora_token), word_vector_length))
    for i in range(len(corpora_token)):
        text_vector = np.zeros(word_vector_length)
        for j in range(len(corpora_token[i])):
            try:
                text_vector += w2v_model[corpora_token[i][j]]
            except:
                pass
            if j == len(corpora_token[i]) - 1:
                text_vector = text_vector / len(corpora_token[i])
        text_matrix[i][:] = text_vector
    return text_matrix
