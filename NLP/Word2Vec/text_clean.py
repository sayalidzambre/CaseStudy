import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


def load_response(dataframe, col_name):
    """
    Function to load survey response from a pandas dataframe into a list
    object that can be passed to the clean_corpus() function
    -----PARAMETERS-----
    dataframe: the pandas dataframe where the survey responses are stored
    col_name: a string of the column name of the survey responses
    -----OUTPUT-----
    Returned object is a list of responses as strings
    """
    try:
        responses = [t[0] for t in dataframe[[col_name]].values.tolist()]
    except (TypeError, NameError):
        print("Please input string as col_name")
        pass
    return responses


def clean_corpus(texts, string_line=True, stopping=True, pos='v'):
    """
    Function to clean up survey answers and return list for NLP processing
    --------PARAMETERS---------
    texts: list objects that contains survey response strings
    string_line: if True, each returned survey response is a single string
    if False, each response is a list of words in the original sequence
    stopping: (default) if True, filter stopwords
    pos: (default) if 'v', lemmatize input words as verbs;
    if 'n', lemmatize input words as nouns
    """
    cleaned = []
    i = 0
    stop = set(stopwords.words("english"))
    # print("$$$ empty cleaned created")
    print(">>>> response cleaning initiated")
    for text in texts:
        if (i + 1) % 500 == 0:
            print("--cleaning response #{} out of {}".format(i + 1, len(texts)))
        try:
            text = re.sub("[^a-zA-Z]", " ", text)
            text = word_tokenize(text)
            text = [t.lower() for t in text]
            if stopping:
                text = [t for t in text if t not in stop]
            lemmatizer = WordNetLemmatizer()
            text = [lemmatizer.lemmatize(t, pos=pos) for t in text]
            # TODO: determine which lemmatizer to use for this project
            cleaned.append(text)
        except TypeError:
            cleaned.append([])
        i += 1
    if string_line:
        cleaned = [" ".join(t) for t in cleaned]
    return cleaned


def get_bow(tokenized_text):
    """
    Function to generate bow_list and word_freq from a tokenized_text
    -----PARAMETER-----
    tokenized_text should be in the form of [['a'], ['a', 'b'], ['b']] format,
    where the object is a list of survey response, with each survey response
    as a list of word tokens
    -----OUTPUT-----
    The function returns two objects
    bow_list: a list of Counter objects with word frequency of each response
    word_freq: a Counter object that summarizes the word frequency of the input
    tokenized_text
    """
    bow_list = []
    word_freq = Counter()
    for text in tokenized_text:
        bow = Counter(text)
        word_freq.update(text)
        bow_list.append(bow)
    print("This corpus has {} key words, and the 10 \
most frequent words are: {}".format(len(word_freq.keys()), word_freq.most_common(10)))
    return bow_list, word_freq
