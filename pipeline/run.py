import pandas as pd
from model_loop import ModelLoop
from sklearn.model_selection import train_test_split
import argparse
import spacy
from transform_features import get_feature_transformer
from collections import defaultdict
import numpy as np

def pipeline(args):
    '''
    Runs the model loop.
    '''
    df = pd.read_csv(args.filename)
    df.loc[:,args.x_label] = df[args.x_label].fillna("None")
    if args.dedupe:
        df = df.drop_duplicates(subset='content')
    if args.reduce:
        df = restrict_sources(df)
    X = df[args.x_label]
    y = df[args.y_label]
    parser = spacy.load('en')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    loop = ModelLoop(X_train, X_test, y_train, y_test, args.models,
                     args.iterations, args.output_dir,
                     thresholds = args.thresholds, ks = args.ks,
                     setting=args.features[0])
    loop.run()

def restrict_sources(df, column='source', max_size=500, random_state=1):
    '''
    Resamples data set such that samples with more than n=500 are re-sampled
    randomly.
    '''
    print("Resampling sources with frequency larger than {}".format(max_size))
    counts = df.groupby(column).count()
    counts['count'] = counts[counts.columns[0]]
    counts = counts.loc[:,'count']
    to_sample = defaultdict(lambda:set([]))
    for row in df.itertuples():
        if counts.loc[row.source] > max_size:
            to_sample[row.source].add(row[0])
    remove = []
    for source in to_sample:
        size = counts.loc[source] - max_size
        remove += list(np.random.choice(list(to_sample[source]), size=size, replace=False))
    df.drop(remove, inplace=True)
    return df

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run a model loop')
    parser.add_argument('filename', type=str,
                    help='Location of data file')
    parser.add_argument('--x_label', type=str,
                    help='Label for text column (X)', default = 'content')
    parser.add_argument('--y_label', type=str,
                    help='Label for outcome column (Y)', default = 'label')
    parser.add_argument('--models', nargs='+',
                    help='Models to run', default = ['LR'])
    parser.add_argument('--iterations', type=int,
                    help='Number of iterations', default = 3)
    parser.add_argument('--thresholds', nargs='+', type=float,
                    help='Thresholds', default = [])
    parser.add_argument('--output_dir', type=str,
                    help='Output directory', default = 'output/')
    parser.add_argument('--dedupe', help="dedupe content column",
                    action="store_true", default = True)
    parser.add_argument('--ks', nargs='+', type=float, help='Metrics at k',
                    default = [0.01, 0.05, 0.10, 0.15, 0.2])
    parser.add_argument('--reduce', nargs=1, type=int, help='Restrict sample size from large sources',
                    default = False)
    parser.add_argument('--features', nargs=1, type=str, help='Restrict features.', default='both_only')

    args = parser.parse_args()
    print(args)
    pipeline(args)
