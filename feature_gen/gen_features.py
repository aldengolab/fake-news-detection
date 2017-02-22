import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import ParameterGrid
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import string
import re
import numpy as np
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from collections import defaultdict
from spacy.en import English
import yaml

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]


def define_params():
    '''
    Dictionary with different parameter combinations to try for each feature.
    '''
    params = {'CountVectorize': {
                 'ngram_range':[(1,1)
#                                 , (1,2)
                               ]},
              'TfidfVectorize': {
                 'ngram_range':[(1,2)]},
              'CountPOS':{
                 'language': ['english']}}

    return params

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    
    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)
    
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    
    # lowercase
    text = text.lower()

    return text

class FeatureGenerator():
    '''
    Generates a set of features given a labeled dataset.
    Creates a reusable pipeline to generate the same features for future unknown examples.
    '''

    def __init__(self, datafile, text_label, y_label, fts_to_try):
        self.text_label = text_label
        self.y_label = y_label
        self.fts_to_try = fts_to_try

        # get features and parameters to try
        self.params = define_params()
        self.parser = English()

        # Read in data
        self.data = pd.read_csv(datafile)
        self.raw_text = self.data[text_label].apply(cleanText)
        self.y_train = self.data[y_label]
        self.y_test = None
        self.pipeline = {}

        # Generate features from raw text
        self.X_train = self.fit()
        self.X_test = None

    def transform(self, new_datafile):
        '''
        Generate features for never before seen data. Saves as attribute to
        X_test and y_test.
        '''
        self.new_data = pd.read_csv(new_datafile)
        self.new_raw_text = self.new_data[self.text_label].apply(cleanText)
        self.y_test = self.new_data[self.y_label]

        X = None

        for step in self.pipeline.keys():
            feat_generator = getattr(self, step)
            x_features = feat_generator(step=step)
            if X != None:
                    X = hstack((X, x_features))
            else:
                X = x_features

        self.X_test = X
        print("{} features generated for {} examples".format((self.X_test.shape)[1], (self.X_test.shape)[0]))


    def fit(self):
        '''
        Generates features for labeled data.
        Saves feature generator objects (e.g. fitted vectorizers)
        for future use with unlabeled data.
        '''
        X = None
        for f in self.fts_to_try:
            print("Creating feature: ",f)
            parameter_values = self.params[f]
            for p in ParameterGrid(parameter_values):
                print("Parameters: ",p)
                feat_generator = getattr(self, f)
                x_features, transformer = feat_generator(p)
                self.pipeline[f] = transformer
                if X != None:
                    X = hstack((X, x_features))
                else:
                    X = x_features
        print("{} features generated for {} examples".format((X.shape)[1], (X.shape)[0]))
        return X
    
    # A custom function to tokenize the text using spaCy
    # and convert to lemmas
    def tokenizeText(self, sample):

        # get the tokens using spaCy
        tokens = self.parser(sample)

        # lemmatize
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas

        # stoplist the tokens
        tokens = [tok for tok in tokens if tok not in STOPLIST]

        # stoplist symbols
        tokens = [tok for tok in tokens if tok not in SYMBOLS]

        # remove large strings of whitespace
        while "" in tokens:
            tokens.remove("")
        while " " in tokens:
            tokens.remove(" ")
        while "\n" in tokens:
            tokens.remove("\n")
        while "\n\n" in tokens:
            tokens.remove("\n\n")

        return tokens

    ### BEGIN FEATURE GEN FUNCTIONS ###
    ### Any of these function names can be added to
    ### the list of features to try in feature_config.yaml.
    ### Additional parameters combinations can be defined in define_params()


    def CountVectorize(self, kwargs=None, step='fit'):
        '''
        Creates a sparse matrix of normalized counts of words from document.
        kwargs are generated from the parameters dictionary
        '''
        if step == 'fit':
            v = CountVectorizer(tokenizer=self.tokenizeText,
                             max_features=3000, **kwargs)
            x_features = v.fit_transform(self.raw_text, self.y_train)
            print("xft size", x_features.shape)
            return x_features, v
        else:
            v = self.pipeline[step]
            x_features = v.transform(self.new_raw_text)
            return x_features


    def TfidfVectorize(self, kwargs=None, step='fit'):
        '''
        Creates a sparse matrix of normalized TFIDF counts of words from document.
        kwargs are generated from the parameters dictionary
        '''
        if step == 'fit':
            v = TfidfVectorizer(tokenizer=self.tokenizeText,
                             max_features=3000, **kwargs)
            x_features = v.fit_transform(self.raw_text, self.y_train)
            return x_features, v
        else:
            v = self.pipeline[step]
            x_features = v.transform(self.new_raw_text)
            return x_features

    def CountPOS(self, kwargs=None, step='fit'):
        '''
        Creates a sparse matrix of part of speech frequencies for each document.
        kwargs are generated from the parameters dictionary
        '''
        def count_pos2(s):
            tagged = self.parser(s)
            counts = defaultdict(int)
            for w in tagged:
                counts[w.dep_] += 1
            for k, v in counts.items():
                counts[k] = counts[k]/sum(counts.values())
            return counts
        
        def count_pos(s):
            tagged = pos_tag(word_tokenize(s))
            counts = defaultdict(int)
            for (word, tag) in tagged:
                counts[tag] += 1
            for k, v in counts.items():
                counts[k] = counts[k]/sum(counts.values())
            return counts

        if step == 'fit':
            v = DictVectorizer()
            d = self.raw_text.apply(count_pos2)
            x_features = v.fit_transform(d)
            return x_features, v
        else:
            v = self.pipeline[step]
            d = self.new_raw_text.apply(count_pos2)
            x_features = v.transform(d)
            return x_features


    
    ### END FEATURE GEN FUNCTIONS ###

if __name__ == "__main__":
    with open("feature_config.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    args = {k: v for k, v in cfg.items() if k != 'test_datafile'}
    f = FeatureGenerator(**args)
    f.transform(cfg['test_datafile'])
