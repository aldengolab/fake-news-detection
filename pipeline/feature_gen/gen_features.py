import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import ParameterGrid
import numpy as np
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from collections import defaultdict
import yaml

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
        
        # Read in data
        self.data = pd.read_csv(datafile)
        self.raw_text = self.data[text_label]
        self.y = self.data[y_label]
        self.pipeline = {}
        
        # Generate features from raw text
        self.X = self.fit()
        
    def transform(self, new_datafile):
        '''
        Generate features for never before seen data.
        '''
        self.new_data = pd.read_csv(new_datafile)
        self.new_raw_text = self.new_data[self.text_label]
        
        X = None
        
        for step in self.pipeline.keys():
            feat_generator = getattr(self, step)
            x_features = feat_generator(step=step)
            if X != None:
                    X = hstack((X, x_features))     
            else:
                X = x_features
        
        self.new_X = X
        print("{} features generated for {} examples".format((self.new_X.shape)[1], (self.new_X.shape)[0]))
            
        
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
            v = CountVectorizer(tokenizer=nltk.word_tokenize,
                             stop_words='english',
                             max_features=3000, **kwargs)
            x_features = v.fit_transform(self.raw_text, self.y)
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
            v = TfidfVectorizer(tokenizer=nltk.word_tokenize,
                             stop_words='english',
                             max_features=3000, **kwargs)
            x_features = v.fit_transform(self.raw_text, self.y)
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
            d = self.raw_text.apply(count_pos)
            x_features = v.fit_transform(d)
            return x_features, v
        else:
            v = self.pipeline[step]
            d = self.new_raw_text.apply(count_pos)
            x_features = v.transform(d)
            return x_features
        
        
    ### END FEATURE GEN FUNCTIONS ###

if __name__ == "__main__":
    with open("feature_config.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    args = {k: v for k, v in cfg.items() if k != 'test_datafile'}
    f = FeatureGenerator(**args)
    print(f.y.shape)
    print(np.arange(1))
    f.transform(cfg['test_datafile'])
