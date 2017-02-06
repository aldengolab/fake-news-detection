# Train-test split code

import pandas as pd
from sklearn import modelselection

class test_train():
    '''
    Provides a suite to test-train split functionality.
    '''

    def __init__(self, label=None, k=None, data_path=None):
        self.data_path = data_path
        self.label = None
        self.data = None
        self.k = None
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        if data_path:
            self.load_data(datapath)

    def load_data(self, datapath):
        '''
        Loads a csv as a pandas dataframe.
        '''
        self.data = pd.read_csv(datapath)

    def split_k(self, k, label, x_variables=None, data=self.data, random=1234, stratification=None):
        '''
        Takes a k value within (0, 1) specifying the percent of the dataset to
        be held for testing. Requires label. Will treat all other columns as
        X variables. Default will use dataset loaded into class. Can
        take lists, numpy arrays, scipy-sparse matrices, and pandas dataframes.
        '''
        assert k < 1 and k > 0
        self.label = label
        self.k = k
        
        y = data[label]
        if x_variables = None:
            X = data[x for x in data.columns where x is not label]
        else:
            X = x_variables

        self.x_train, self.s_test, self.y_train, self.y_test = \
                        modelselection.test_train_split(X, y, test_size = k,
                                                        random_state=random,
                                                        stratify=stratification)

    def split_time(self, max_train_date, data=self.data):
        '''
        Takes a max_train_date value after which values will only be used for
        the test set. Defaults to using loaded data.
        '''
        pass

    def progressive_time_split(self, train_period, test_period, data=self.data):
        '''
        Uses the period to split the data into individual periods on which to
        execute the split_time function. Splits data into chunks of train_period
        + test_period length, then splits those chunks into test/train sets.
        '''
        pass
