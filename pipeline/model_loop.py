# Loops through ML models for classification

# Basic code borrowed from RAYID GHANI, with extensive edits.
## https://github.com/rayidghani/magicloops/blob/master/magicloops.py

from __future__ import division
import pandas as pd
import numpy as np
import random
import os
from scipy.sparse import isspmatrix_csc, csc_matrix
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *
import spacy
import traceback
from transform_features import get_feature_transformer


from model import Model

class ModelLoop():

    def __init__(self, X_train, X_test, y_train, y_test, models, iterations, output_dir,
                 thresholds = [.1], ks = [], ignore_columns=[], method='pandas',
                 report='full', pickle=False, roc=False, unique_identifiers=['id'],
                 setting='both_only'):
        '''
        Constructor for the ModelLoop.

        Inputs:
         - train: training data as a pandas dataframe
         - test: testing data as a pandas dataframe
         - models: models to run as list
         - iterations: maximum number of parameter iterations as int
         - output_dir: directory output model performance
         - report: type of reporting, options are simple and full
         - pickle: whether to pickle models
        '''
        self.raw_X_train = X_train
        self.raw_X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models_to_run = models
        self.iterations_max = iterations
        self.output_dir = output_dir
        self.params_iter_max = 50
        self.thresholds = thresholds
        self.ks = ks
        if self.thresholds == [] and self.ks == []:
            raise NameError('Either thresholds or ks must contain values.')
        self.method = method
        self.clfs = None
        self.params = None
        self.ignore = ignore_columns
        self.X_variables = []
        self.define_clfs_params()
        self.report = report
        assert (report == 'simple' or report == 'full')
        self.pickle = pickle # Not currently supported
        self.roc = roc
        self.unique_identifiers = unique_identifiers
        self.setting = setting
        if self.setting not in ['both_only', 'all', 'tfidf_only', 'grammar_only']:
            raise NameError('Incorrect feature setting.')

    def define_clfs_params(self):
        '''
        Defines all relevant parameters and classes for classfier objects.
        Edit these if you wish to change parameters.
        '''
        # These are the classifiers
        self.clfs = {
            'RF': RandomForestClassifier(n_estimators = 50, n_jobs = -1),
            'ET': ExtraTreesClassifier(n_estimators = 10, n_jobs = -1, criterion = 'entropy'),
            'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth = [1, 5, 10, 15]), algorithm = "SAMME", n_estimators = 200),
            'LR': LogisticRegression(penalty = 'l1', C = 1e5),
            'SVM': svm.SVC(kernel = 'linear', probability = True, random_state = 0),
            'GB': GradientBoostingClassifier(learning_rate = 0.05, subsample = 0.5, max_depth = 6, n_estimators = 10),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(),
            'SGD': SGDClassifier(loss = 'log', penalty = 'l2'),
            'KNN': KNeighborsClassifier(n_neighbors = 3)
            }
        # These are the parameters which will be run through
        self.params = {
             'RF':{'n_estimators': [1,10,100,1000], 'max_depth': [10, 15,20,30,40,50,60,70,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
             'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'random_state': [1]},
             'SGD': {'loss': ['log'], 'penalty': ['l2','l1','elasticnet'], 'random_state': [1]},
             'ET': {'n_estimators': [1,10,100,1000], 'criterion' : ['gini', 'entropy'], 'max_depth': [1,3,5,10,15], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
             'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000], 'random_state': [1]},
             'GB': {'n_estimators': [1,10,100,1000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100], 'random_state': [1]},
             'NB': {},
             'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,2,15,20,30,40,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
             'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear'], 'random_state': [1]},
             'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
             }

    def clf_loop(self, X_train, X_test, y_train, y_test, individuals, setting):
        '''
        Runs through each model specified by models_to_run once with each possible
        setting in params.
        '''
        N = 0
        self.prepare_report()
        for index, clf in enumerate([self.clfs[x] for x in self.models_to_run]):
            iteration = 0
            print('Running {}.'.format(self.models_to_run[index]))
            parameter_values = self.params[self.models_to_run[index]]
            grid = ParameterGrid(parameter_values)
            while iteration < self.iterations_max and iteration < len(grid):
                print('    Running Iteration {} of {}...'.format(iteration + 1, self.iterations_max))
                if len(grid) > self.iterations_max:
                    p = random.choice(list(grid))
                else:
                    p = list(grid)[iteration]
                try:
                    m = Model(clf, X_train, y_train, X_test, y_test, p, N,
                                   self.models_to_run[index], iteration,
                                   self.output_dir, thresholds = self.thresholds,
                                   ks = self.ks, report = self.report, label='label',
                                   individuals=individuals, setting=setting)
                    m.run()
                    print('    Printing to file...')
                    if not self.roc:
                        m.performance_to_file()
                    else:
                        m.performance_to_file(roc='{}ROC_{}_{}-{}.png'.format(
                            self.output_dir, self.models_to_run[index], N,
                            iteration))
                except IndexError as e:
                    print(p)
                    print(N)
                    print('IndexError: {}'.format(e))
                    print(traceback.format_exc())
                    continue
                except RuntimeError as e:
                    print(p)
                    print(N)
                    print('RuntimeError: {}'.format(e))
                    print(traceback.format_exc())
                    continue
                except AttributeError as e:
                    print(p)
                    print(N)
                    print('AttributeError: {}'.format(e))
                    print(traceback.format_exc())
                    continue
                iteration += 1
            N += 1
    def prepare_report(self):
        '''
        Prepares the output file(s).
        '''
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        with open(self.output_dir + '/evaluations.csv', 'w') as f:
            if self.thresholds != []:
                measure = 'threshold'
            else:
                measure = 'k'
            f.write('model_id, model_type, iteration, auc, {}, precision, recall, accuracy, params\n'.format(measure))

    def data_checks(self, dataframe):
        '''
        Checks that data are all present and there are no infinite values.
        '''
        if self.method == 'pandas':

            # Feature generation assumes that each article text is unique.
            assert (self.raw_X_train.duplicated().sum() == 0)
            assert (self.raw_X_test.duplicated().sum() == 0)
            # Remove any infinities, replace with missing
            dataframe=dataframe.replace([np.inf, -np.inf], np.nan)
            # Find any columns with missing values
            missing_cols = []
            for column in self.X_variables:
                assert max(dataframe[column] <= 1)
                if len(dataframe[dataframe[column].isnull()]) > 0:
                    missing_cols.append(column)
            if len(missing_cols) > 0:
                raise NameError('Missing or infinite X values detected: {}'.format(missing_cols))

    def run(self):
        '''
        Loads data from csvs, executes basic data checks, runs loop.

        If roc is not False, will print ROC to the filename specified.
        '''
        # Run Data checks
        if self.method == 'pandas':
            self.data_checks(self.raw_X_train)
            self.data_checks(self.raw_X_test)
        if self.method == 'csc':
            if not isspmatrix_csc(self.raw_X_train):
                self.X_train = csc_matrix(self.raw_X_train)
                self.X_test = csc_matrix(self.raw_X_test)
                self.y_test = csc_matrix(self.y_test)
                self.y_train = csc_matrix(self.y_train)

        individuals = self.y_test

        # Generate features
        parser = spacy.load('en')
        if self.setting == 'all':
            params = [(True, False, 'no_tfidf'), (True, True, 'both'), (False, True, 'no_grammar')]
        if self.setting == 'both_only':
            params = [(True, True, 'both')]
        if self.setting == 'grammar_only':
            params = [(True, False, 'no_tfidf')]
        if self.setting == 'tfidf_only':
            params = [(False, True, 'no_grammar')]
        for params in params:
            print('Feature generation set to {} for this run'.format(params[2]))
            f = get_feature_transformer(parser, run_grammar=params[0], run_tfidf=params[1])
            self.X_train = f.fit_transform(self.raw_X_train).todense()
            self.X_test = f.transform(self.raw_X_test).todense()

            # Run the loop
            self.clf_loop(self.X_train, self.X_test, self.y_train, self.y_test, individuals, params[2])
