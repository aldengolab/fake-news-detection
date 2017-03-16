
# A class to hold models

import numpy as np
import scipy.sparse as sps
import random
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
from transform_features import get_feature_transformer
import pdb
import os


class Model():

    def __init__(self, clf, X_train, y_train, X_test, y_test, p, N, model_type,
                 iteration, output_dir, individuals, setting, label,
                 report='full', threshold=0.75, thresholds=[], ks=[], k=.05):
        '''
        Constructor.
        '''
        self.clf = clf
        # self.X_train = X_train.todense()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        # self.X_test = X_test.todense()
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
        self.output_dir = output_dir
        self.report = report
        self.uniques = individuals
        self.setting = setting
        self.label=label

    def run(self):
        '''
        Runs a model with params p.
        '''
        self.clf.set_params(**self.params)
        # f = get_feature_transformer(self.parser)
        # self.X_train_fts = f.fit_transform(self.X_train)
        # self.X_test_fts = f.transform(self.X_test)
        self.pipeline = Pipeline([
            # ('feature_gen', f),
            ('clf', self.clf),
        ])
        self.y_pred_probs = self.pipeline.fit(self.X_train,self.y_train).predict_proba(self.X_test)[:,1]
        if self.model_type in ['RF', 'ET', 'AB', 'GB', 'DT']:
            self.importances = self.clf.feature_importances_
        elif self.model_type in ['SVM', 'LR', 'SGD']:
            self.importances = self.clf.coef_[0]

    def calc_performance(self, measure):
        '''
        Stores performance given a threshold for prediction.
        '''
        self.roc_auc = self.auc_roc(self.y_test, self.y_pred_probs)
        if self.ks == [] and self.thresholds != []:
            self.threshold = measure
            self.precision = self.precision_at_threshold(self.y_test, self.y_pred_probs, self.threshold)
            self.recall = self.recall_at_threshold(self.y_test, self.y_pred_probs, self.threshold)
            self.accuracy = self.accuracy_at_threshold(self.y_test, self.y_pred_probs, self.threshold)
        elif self.ks != []:
            self.k = measure
            self.precision = self.precision_at_k(self.y_test, self.y_pred_probs, self.k)
            self.recall = self.recall_at_k(self.y_test, self.y_pred_probs, self.k)
            self.accuracy = self.accuracy_at_k(self.y_test, self.y_pred_probs, self.k)

    def performance_to_file(self, pickle_path='Not enabled'):
        '''
        Write results to file.
        '''
        self.pickle_path = pickle_path
        if self.thresholds != []:
            measures = self.thresholds
        elif self.ks != []:
            measures = self.ks
        for measure in measures:
            self.calc_performance(measure)
            self.model_performance_to_file(measure=measure)
        if self.report == 'full':
            if not os.path.isfile(self.output_dir + '/models.csv'):
                self.prepare_files()
            self.model_meta_to_file()
            if self.iteration == 0:
                self.model_group_to_file()
            self.unit_level_to_file()
            self.feature_importances_to_file()

    def model_meta_to_file(self, filename='/models.csv', method='a'):
        '''
        Writes meta information about model, including pickle path, to file.

        COLUMNS
        model_id,label,model_group_id,model_type,model_parameters,pickle_file_path_name
        '''
        with open(self.output_dir + filename, method) as f:
            result = '"{N}-{iteration}", "{label}", "{N}", "{model_type}", "{params}", "{pickle}"\n'.format(
                N = self.N, iteration = self.iteration, label=self.label,
                model_type=self.model_type, params=self.params,
                pickle=self.pickle_path)
            f.write(result)

    def model_performance_to_file(self, measure, filename='/evaluations.csv', method='a'):
        '''
        Writes standard performance metrics (AUC, precision, etc.) to file.

        COLUMNS:
        model_id, label, model_type, iteration, auc, measure, precision, recall, accuracy, params
        '''
        with open(self.output_dir + filename, method) as f:
            result = '"{0}-{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", "{8}", "{9}", "{10}", "{11}"\n'.format(
                self.N, self.iteration, self.label, self.setting,
                self.model_type, self.iteration, self.roc_auc, measure,
                self.precision, self.recall, self.accuracy, self.params)
            f.write(result)

    def model_group_to_file(self, filename='/model_groups.csv', method='a'):
        '''
        Writes model group information to file.

        COLUMNS:
        model_group_id,model_type,feature_list
        '''
        with open(self.output_dir + filename, method) as f:
            result = '"{0}","{1}","{2}"\n'.format(
                self.N, self.model_type, self.setting)
            f.write(result)

    def unit_level_to_file(self, pred_file='/predictions.csv', feat_file='/individual_importances.csv', method='a', date_col='begin_date'):
        '''
        Writes unit level reporting to file, namely unit predictions & scores
        plus feature importances for each unit (x_vector * importance).

        COLUMNS FOR individual_importances:
        'model_id','article_id
        COLUMNS FOR predictions:
        'model_id,article_id,score,predicted_label,actual_label
        '''
        i = 0
        with open(self.output_dir + pred_file, method) as pred_f:
            for i in range(len(self.uniques.index)):
                unique_id = self.uniques.index[i]
                probability = self.y_pred_probs[i]
                label = self.y_test.iloc[i]
                pred_result = '"{0}-{1}","{2}","{3}","{4}"'.format(
                    self.N, self.iteration, unique_id, probability, label)
                pred_f.write(pred_result + '\n')

    def feature_importances_to_file(self, filename='/feature_importances.csv', method='a'):
        '''
        Writes feature importances to file.

        COLUMNS:
        model_id,feature,feature_importance
        '''
        with open(self.output_dir + filename, method) as f:
            for i in range(self.X_train.shape[1]):
                feature = i
                importance = self.importances[i]
                result = '"{0}-{1}", "{2}", "{3}"\n'.format(self.N,
                    self.iteration, feature, importance)
                f.write(result)

    def prepare_files(self):
        '''
        Writes columns for full reportin.
        '''
        with open(self.output_dir + '/models.csv', 'w') as f:
            f.write('model_id,label,model_group_id,model_type,model_parameters,pickle_file_path\n')
        with open(self.output_dir + '/model_groups.csv', 'w') as f:
            f.write('model_group_id,model_type,feature_list\n')
        with open(self.output_dir + '/predictions.csv', 'w') as f:
            f.write('model_id,article_id,score,actual_label\n')
        with open(self.output_dir + '/feature_importances.csv', 'w') as f:
            f.write('model_id,feature,feature_importance\n')
        with open(self.output_dir + '/individual_importances.csv', 'w') as f:
            headers = ['model_id','article_id'] + [str(i) for i in range(self.X_train.shape[1])]
            f.write(','.join(headers) + '\n')

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
