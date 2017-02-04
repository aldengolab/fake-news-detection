## This code loops through ML models for classification.

## CODE STRUCTURE LIBERALLY BORROWED FROM RAYID GHANI, WITH EXTENSIVE EDITS ##
## https://github.com/rayidghani/magicloops/blob/master/magicloops.py ##
## Accessed: 5/5/2016 ##

from __future__ import division
import sys
import Read
import matplotlib.cm as cm
import copy
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import acg_process
import random

PARAMS_ITER_MAX = 50

def define_project_params(label, models):
    '''
    Parameters specific to the project being run.
    '''
    models_to_run = models
    y_variable = label
    imp_cols = []
    thresholds = [0.5, 0.75, 0.8, 0.9, 0.95]
    robustscale_cols = []
    scale_columns = []

    return (y_variable, imp_cols, models_to_run, robustscale_cols, scale_columns)

def define_clfs_params():
    '''
    Defines all relevant parameters and classes for classfier objects.

    These should be edited for randomized options.
    '''
    # These are the classifiers
    clfs = {
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
    params = {
        'RF':{'n_estimators': [1,10,100,1000], 'max_depth': [10, 15,20,30,40,50,60,70,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
        'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10],'random_state': [1]},
        'SGD': {'loss': ['log','perceptron'], 'penalty': ['l2','l1','elasticnet'], 'random_state': [1]},
        'ET': {'n_estimators': [1,10,100,1000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,3,5,10,15], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
        'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000], 'random_state': [1]},
        'GB': {'n_estimators': [1,10,100,1000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100], 'random_state': [1]},
        'NB' : {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [15,20,30,40,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear'], 'random_state': [1]},
        'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
        }
    return clfs, params

def clf_loop(train, test, clfs, models_to_run, params, y_variable, X_variables,
 imp_cols = [], stat_k = .10, plot = False,
 robustscale_cols = [], scale_columns = []):
    '''
    Runs through each model specified by models_to_run once with each possible
    setting in params. For boosting don't include randomize_features.
    '''
    N = 0
    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        print('Running {}.'.format(models_to_run[index]))
        parameter_values = params[models_to_run[index]]
        grid = ParameterGrid(parameter_values)
        iteration = 0
        while iteration < PARAMS_ITER_MAX:
            p = random.choice(list(grid))
            try:
                run_model(p)
                iteration += 1
            except IndexError as e:
                print('Error: {0}'.format(e))
                continue
            except RuntimeError as e:
                print('RuntimeError: {}'.format(e))
                continue
            except AttributeError as e:
                print('AttributeError: {}'.format(e))
                continue

def run_model(p):
    '''
    Runs a model with params p.
    '''
    clf.set_params(**p)
    print(clf)
    y_pred_probs = clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    result = precision_at_k(y_test, y_pred_probs,
    stat_k)
    print('Precision: ', result)
    if result[0] > maximum[1]:
        maximum = (clf, result[0], result[1])
        print('Max Precision: {}'.format(maximum))
        plot_precision_recall_n(y_test,
        y_pred_probs, clf, N)
        N += 1
        if models_to_run[index] == 'RF':
            importances = clf.feature_importances_
            sortedidx = np.argsort(importances)
            best_features = X_train.columns[sortedidx]
            print(best_features[::-1])
        if models_to_run[index] == 'DT':
            export_graphviz(clf, 'DT_graph_' + str(N) + '.dot')

    result = auc_roc(y_test, y_pred_probs)
    print('AUC: {}'.format(result))
    plot_precision_recall_n(y_test, y_pred_probs, clf, N)
    N += 1
    print('Recall: ', recall_at_k(y_test, y_pred_probs, stat_k))
    print('Accuracy: ', accuracy_at_k(y_test, y_pred_probs, stat_k))


def plot_precision_recall_n(y_true, y_prob, model_name, N):
    '''
    Plots the precision recall curve.
    '''
    from sklearn.metrics import precision_recall_curve

    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)

    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')

    model = str(model_name)[:20]
    name = '{}_{}.png'.format(model, N)
    print('File saved as {} for model above'.format(name))
    plt.title(name)
    plt.savefig(name)
    plt.close()

def accuracy(y_true, y_scores, threshold):
    '''
    Dyanamic k-threshold accuracy. Defines threshold for Positive at the
    value that returns the k*n top values where k is within [0-1]. If k is not
    specified, threshold will default to THRESHOLD.
    '''
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.accuracy_score(y_true, y_pred), threshold)

def auc_roc(y_true, y_scores):
    '''
    Computes the Area-Under-the-Curve for the ROC curve.
    '''
    return metrics.roc_auc_score(y_true, y_scores)

def precision_at_threshold(y_true, y_scores, threshold):
    '''
    Dyanamic k-threshold precision. Defines threshold for Positive at the
    value that returns the k*n top values where k is within [0-1]. If k is not
    specified, threshold will default to THRESHOLD.
    '''
    if k != None:
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    else:
        threshold = THRESHOLD
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.precision_score(y_true, y_pred), threshold)

def recall_at_threshold(y_true, y_scores, threshold):
    '''
    Dyanamic k-threshold recall. Defines threshold for Positive at the
    value that returns the k*n top values where k is within [0-1].
    '''
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.recall_score(y_true, y_pred), threshold)

def run_data_checks(dataframe, y_variable, X_variables):
    '''
    Checks that data are all present.
    '''
    # Remove any infinities, replace with missing
    dataframe=dataframe.replace([np.inf, -np.inf], np.nan)
    # Find any columns with missing values
    missing_cols = []
    for column in X_variables:
        assert max(dataframe[column] <= 1)
        if len(dataframe[dataframe[column].isnull()]) > 0:
            missing_cols.append(column)
    if len(missing_cols) > 0:
        raise NameError('Missing or infinite X values detected: {}'.format(missing_cols))
    if len(dataframe[y_variable].notnull()) < dataframe[y_variable]:
        raise NameError('Missing or infinite labels detected.')

def main(trainfp, testfp, label, models, iterations):
    '''
    Loads data from csvs, executes basic data checks, runs loop.
    '''
    train = pd.read_csv(trainfp)
    test = pd.read_csv(testfp)
    assert set(train.columns) == set(test.columns)

    # Get all the necessary parameters
    clfs, params = define_clfs_params(label, models)
    y_variable, imp_cols, models_to_run, robustscale_cols, scale_columns = define_project_params()
    X_variables = [col for col in train.columns if col != y_variable]

    # Run Data checks
    print("Checking training data...")
    run_data_checks(train, y_variable, X_variables)
    print("Checking testing data...")
    run_data_checks(train, y_variable, X_variables)

    # Run the loop
    clf_loop(test, train, clfs, models_to_run, params, y_variable, X_variables,
        imp_cols = imp_cols, scale_columns = scale_columns, robustscale_cols = robustscale_cols)

if __name__ == '__main__':
    arser = argparse.ArgumentParser(description='A model loop for ML pipelines.)
    parser.add_argument('--train', nargs=1, help='Filepath for csv containing training data')
    parser.add_argument('--test', nargs=1, help='Filepath for csv containing test data')
    parser.add_argument('--label', nargs=1, help='Label column for prediction')
    parser.add_argument('--models', nargs='*',
                         help='List of models to run: RF, AB, ET, LR, SVM, GB, NB, DT, SGD, KNN included')
    parser.add_argument('-n', nargs=1, help='Number of iterations to run')
    parser.add_argument('-o', nargs=1, help='Output CSV' default='output.csv')
    args = parser.parse_args()
    main(args.train[0], args.test[0], args.label[0], args.models, args.n[0])
