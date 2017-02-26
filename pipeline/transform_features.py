import yaml
import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import collections

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]

class GrammarTransformer():
    """
    Convert text to counts of syntactic structure
    """
    def __init__(self, parser):
        self.parser = parser
        
    def transform(self, X, y=None,**transform_params):
        print('here')
        return countgrammart(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    
    def countgrammar(self, texts):
        lookup = {}
        for i, x in enumerate(texts):
            lookup[x] = i
        grammar_counts = {}
        for doc in self.parser.pipe(texts, batch_size=1000, n_threads=8):
            counts = collections.Counter()
            for w in doc:
                counts[w.dep_] += 1

            for k, v in counts.items():
                counts[k] = counts[k]/sum(counts.values())
            #doc.user_data['my_data'] = counts
            grammar_counts[doc.text] = counts
        print('here3')
        rv = list(range(len(texts)))
        for text, i in lookup.items():
            rv[i] = grammar_counts[text]
        return rv

class PreProcessor():
    def __init__(self, parser):
        self.parser = parser

    def transform(self, X, y=None, **transform_params):
        print('here')
        return self.tokenizeText(X)

    def fit(self, X, y=None, **fit_params):
        return self.tokenizeText(X)

    def get_params(self, deep=True):
        return {}

    def tokenizeText(self, texts):
        lookup = {}
        for i, x in enumerate(texts):
            lookup[x] = i
        token_dict = {}
        for doc in self.parser.pipe(texts, batch_size=1000, n_threads=8):
            # tokens = self.parser(s)
            # print(doc)
            lemmas = []
            # If a token is a named entity, replace it with <NAME>, <PLACE> etc
            for tok in doc:
                lemmas.append(tok.lemma_.lower().strip() if tok.ent_type_ == "" else "<{}>".format(tok.ent_type_))
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

            token_dict[doc.text] = tokens
        print('here3')
        rv = list(range(len(texts)))
        for text, i in lookup.items():
            rv[i] = token_dict[text]
        return rv

class TFIDFTransformer():
    """
    Convert text to vectorized TFIDF counts, 
    Named entities are replaced with anonymous tokens
    """
    def __init__(self, parser):
        self.parser = parser
        # self.vectorizer = TfidfVectorizer(
        #                  max_features=3000)
        
    def transform(self, X, y=None, **transform_params):
        print('here')
        return self.vectorizer.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self.vectorizer.fit(X)

    def get_params(self, deep=True):
        return {}

    def tokenizeText(self, tokens):
        # tokens = self.parser(s)
        # print(tokens)
        lemmas = []
        # If a token is a named entity, replace it with <NAME>, <PLACE> etc
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.ent_type_ == "" else "<{}>".format(tok.ent_type_))
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

def get_feature_transformer(parser):

    tfidf = TFIDFTransformer(parser=parser)
    tfidf = Pipeline([
            ('pre', PreProcessor(parser=parser)),
            ('vect', TfidfVectorizer(
                         max_features=3000)),
            ('clf', None)
        ])
    grammar_counter = Pipeline([
            ('grm', GrammarTransformer(parser=parser)),
            ('to_dict', DictVectorizer()),
            ('clf', None)
        ])
    feature_transformer = FeatureUnion([("tfidf", tfidf), ('grammar_counter', grammar_counter)])
    return feature_transformer