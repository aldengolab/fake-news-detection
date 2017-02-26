import pandas as pd
from model_loop import ModelLoop
from sklearn.model_selection import train_test_split


df = pd.read_csv('../data_cleaning/articles1.csv')
X = df['content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y)
# label = 'LABEL'
models = ['DT','LR']
iterations = 2
output_dir = 'output/'
loop = ModelLoop( X_train, X_test, y_train, y_test, models, iterations, output_dir)
loop.run()