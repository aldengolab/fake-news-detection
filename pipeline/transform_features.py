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
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”"]

class GrammarTransformer():
    """
    Convert text to counts of syntactic structure
    """
    def __init__(self, parser):
        self.parser = parser
        
    def transform(self, X, y=None,**transform_params):
        return self.countgrammar(X)

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
            grammar_counts[doc.text] = counts
        rv = list(range(len(texts)))
        for text, i in lookup.items():
            try:
                rv[i] = grammar_counts[text]
            except:
                # Ocassionally the way Spacey processes unusual characters (bullet points, em dashes) will cause the lookup based on the original characters to fail.
                # In that case, just set to None.
                print("error")
                print(text)
                rv[i] = {}
                continue

        return rv

class PreTokenizer():
    '''
    Custom function to clean text before tokenizing. Uses Spacy entity recognition to replace all mentions of named entities with a placeholder, e.g. <NAME> or <PLACE> etc.
    '''
    def __init__(self, parser):
        self.parser = parser

    def transform(self, X, y=None, **transform_params):
        '''
        X can be an any iterable - array, pandas dataframe, list, etc.
        '''
        return self.tokenizeText(X)

    def fit(self, X, y=None, **fit_params):
        #SkLearn Pipeline expects all transformers to have a fit and tranform method, but we are only using transform.
        return self

    def get_params(self, deep=True):
        return {}

    def tokenizeText(self, texts):
        lookup = {}
        for i, x in enumerate(texts):
            lookup[x] = i
        token_dict = {}
        for doc in self.parser.pipe(texts, batch_size=1000, n_threads=8):
            lemmas = []
            # If a token is a named entity, replace it with <NAME>, <PLACE> etc
            for tok in doc:
                lemmas.append(tok.text.lower().strip() if tok.ent_type_ == "" else "<{}>".format(tok.ent_type_))
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
            t = " ".join(tokens) 
            token_dict[doc.text] = t
        rv = list(range(len(texts)))
        for text, i in lookup.items():
            try:
                rv[i] = token_dict[text]
            except:
                # Ocassionally the way Spacey processes unusual characters (bullet points, em dashes) will cause the lookup based on the original characters to fail.
                # In that case, just set to None.
                rv[i] = "None"
                continue
        return rv


class CleanTextTransformer():
    """
    Transformer object to convert text to cleaned text.
    """

    def transform(self, X, **transform_params):
        '''
        X can be an any iterable - array, pandas dataframe, list, etc.
        '''
        return [self.cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        #SkLearn Pipeline expects all transformers to have a fit and tranform method, but we are only using transform.
        return self

    def get_params(self, deep=True):
        return {}
    
    # A custom function to clean the text before sending it into the vectorizer
    def cleanText(self, text):
        # text = text.translate({0x2014: None})
        # get rid of newlines
        text = text.strip().replace("\n", " ").replace("\r", " ")
        
        # replace twitter @mentions
        mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
        text = mentionFinder.sub("@MENTION", text)

        # replace emails @mentions
        emailFinder = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
        text = emailFinder.sub("<EMAIL>", text)
        
        # replace HTML symbols
        text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
        # for s in SYMBOLS:
        #     text = text.replace(s, '')
        return text


def get_feature_transformer(parser):
    '''
    Creates a transformer object that will take a text series and generate TFIDF counts and frequency of syntactical structures. 
    Suitable for use as a step in a SKLearn Pipeline.

    inputs: 
        parser: a Spacy pipeline object
    returns:
        feature transformer: FeatureUnion
    '''
    # tfidf = TFIDFTransformer(parser=parser)
    tfidf = Pipeline([
            ('cln', CleanTextTransformer()),
            ('pre', PreTokenizer(parser=parser)),
            ('vect', TfidfVectorizer(
                         max_features=3000)),
            ('clf', None)
        ])
    grammar_counter = Pipeline([
            ('cln', CleanTextTransformer()),
            ('grm', GrammarTransformer(parser=parser)),
            ('to_dict', DictVectorizer()),
            ('clf', None)
        ])
    feature_transformer = FeatureUnion([("tfidf", tfidf), ('grammar_counter', grammar_counter)])
    return feature_transformer