# Model learning pipeline

## Sample Run

`from test_train import TimeTestTrainSplit`  
`from model_loop import ModelLoop`

`label = 'LABEL'`  
`models = ['NB', 'RF', 'ET', 'LR', 'SVM']`  
`iterations = 10`  
`output_dir = 'output/'`  
`for test, train in splits:`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`loop = ModelLoop(train, test, label, models, iterations, output_dir, ignore_columns = ['DATE'])`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`loop.run()`  
`pd.read_csv('output/simple_report.csv', quotechar='"', skipinitialspace = True)`
