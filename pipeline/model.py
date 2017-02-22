
# A class to hold models

import numpy as np
import scipy.sparse as sps
import random
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt

class Model():

    def __init__(self, clf, X_train, y_train, X_test, y_test, p, N, model_type,
                 iteration, output_dir, report='simple',
                 threshold=0.75, thresholds=[], ks=[], k=.05):
        '''
        Constructor.
        '''
        self.clf = clf
        self.X_train = X_train.todense()
        self.y_train = y_train
        self.X_test = X_test.todense()
        self.y_test = y_test
        self.params = p
        self.N = N
        self.model_type = model_type
        self.iteration = iteration
        self.thresholds = thresholds
        self.threshold = threshold
        self.k = k
        self.ks = ks
        self.y_pred_probs = None
        self.roc_auc = None
        self.precision = None
        self.recall = None
        self.accuracy = None
        self.best_features = None
        self.output_dir = output_dir
        self.report = report

    def run(self):
        '''
        Runs a model with params p.
        '''
        self.clf.set_params(**self.params)
        self.y_pred_probs = self.clf.fit(self.X_train,self.y_train).predict_proba(self.X_test)[:,1]
        '''
        if self.model_type is 'RF':
            importances = self.clf.feature_importances_
            sortedidx = np.argsort(importances)
            self.best_features = self.X_train.columns[sortedidx]
        if self.model_type is 'DT':
            export_graphviz(self.clf, 'DT_graph_' + str(self.N) + '.dot')
        '''
    def calc_performance(self, threshold):
        '''
        Stores performance given a threshold for prediction.
        '''
        self.threshold = threshold
        self.roc_auc = self.auc_roc(self.y_test, self.y_pred_probs)
        if self.ks == [] and self.thresholds != []:
            self.precision = self.precision_at_threshold(self.y_test, self.y_pred_probs, self.threshold)
            self.recall = self.recall_at_threshold(self.y_test, self.y_pred_probs, self.threshold)
            self.accuracy = self.accuracy_at_threshold(self.y_test, self.y_pred_probs, self.threshold)
        elif self.ks != []:
            self.precision = self.precision_at_k(self.y_test, self.y_pred_probs, self.k)
            self.recall = self.recall_at_k(self.y_test, self.y_pred_probs, self.k)
            self.accuracy = self.accuracy_at_k(self.y_test, self.y_pred_probs, self.k)

    def performance_to_file(self, roc=False):
        '''
        Write results to file.

        If roc is not False, will print ROC to the filename specified.
        '''
        if self.thresholds != []:
            measures = self.thresholds
        elif self.ks != []:
            measures = self.ks

        for measure in measures:
            self.calc_performance(measure)
            if roc:
                self.print_roc(self.y_test, self.y_pred_probs, roc)
            if self.report == 'simple':
                with open(self.output_dir + 'simple_report.csv', 'a') as f:
                    result = '"{}-{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}"\n'.format(
                        self.N, self.iteration, self.model_type, self.iteration,
                        self.roc_auc, measure, self.precision,
                        self.recall, self.accuracy, self.params)
                    f.write(result)

    def auc_roc(self, y_true, y_scores):
        '''
        Computes the Area-Under-the-Curve for the ROC curve.
        '''
        return metrics.roc_auc_score(y_true, y_scores)

    def accuracy_at_threshold(self, y_true, y_scores, threshold):
        '''
        Dyanamic threshold accuracy.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
        return metrics.accuracy_score(y_true, y_pred)

    def precision_at_threshold(self, y_true, y_scores, threshold):
        '''
        Dyanamic threshold precision.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
        return metrics.precision_score(y_true, y_pred)

    def recall_at_threshold(self, y_true, y_scores, threshold):
        '''
        Dyanamic threshold recall.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
        return metrics.recall_score(y_true, y_pred)

    def recall_at_k(self, y_true, y_scores, k):
        '''
        Dynamic k recall, where 0<k<1.
        '''
        y_pred = self.k_predictions(y_scores, k)
        return metrics.recall_score(y_true, y_pred)

    def precision_at_k(self, y_true, y_scores, k):
        '''
        Dynamic k precision, where 0<k<1.
        '''
        y_pred = self.k_predictions(y_scores, k)
        return metrics.precision_score(y_true, y_pred)

    def accuracy_at_k(self, y_true, y_scores, k):
        '''
        Dynamic k accuracy, where 0<k<1.
        '''
        y_pred = self.k_predictions(y_scores, k)
        return metrics.accuracy_score(y_true, y_pred)

    def k_predictions(self, y_scores, k):
        '''
        Returns the y_pred vector as a numpy array using 0<k<1 as the prediction
        threshold.
        '''
        y_scores = list(enumerate(y_scores))
        y_scores.sort(key=lambda x: x[1])
        cut = np.floor(len(y_scores) * (1-k))
        y_pred = [0] * len(y_scores)
        for i in range(len(y_scores)):
            if i >= cut:
                y_pred[i] = (y_scores[i][0], 1)
            else:
                y_pred[i] = (y_scores[i][0], 0)
        y_pred.sort()
        rv = []
        for y in y_pred:
            rv.append(y[1])
        return np.asarray(rv)

    def print_roc(self, y_true, y_scores, filename):
        '''
        Prints the ROC for this model.
        '''
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % self.roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(filename)
        plt.close()
