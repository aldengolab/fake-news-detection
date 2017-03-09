import pandas as pd
from model_loop import ModelLoop
from sklearn.model_selection import train_test_split
import argparse
import spacy
from transform_features import get_feature_transformer

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
                    help='Thresholds', default = [0.9, 0.8, 0.75,  0.7, 0.6])
    parser.add_argument('--output_dir', type=str,
                    help='Output directory', default = 'output/')
    parser.add_argument('--dedupe', help="dedupe content column",
                    action="store_true")
    parser.add_argument('--ks', nargs='+', type=float, help='Metrics at k',
                    default = [])

    args = parser.parse_args()
    print(args)

    df = pd.read_csv(args.filename)
    if args.dedupe:
        df = df.drop_duplicates(subset='content')
    # print(df.head())
    X = df[args.x_label]
    # print(X.head())
    y = df[args.y_label]
    # print(y.head())
    parser = spacy.load('en')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    loop = ModelLoop(X_train, X_test, y_train, y_test, args.models,
                     args.iterations, args.output_dir, 
                     thresholds = args.thresholds, ks = args.ks)
    loop.run()
