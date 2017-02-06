# fake-news-detection
Final Project for CAPP 30255 Advanced Machine Learning for Public Policy
Lauren Dyson & Alden Golab, Winter 2017

# Dependencies
- python 3.x
- nltk
- numpy
- sklearn
- scipy
- pandas

# Data

This project is using a [dataset published by Signal Media](http://research.signalmedia.co/newsir16/signal-dataset.html) in conjunction with the Recent Trends in News Information Retrieval 2016 conference to facilitate conducting research on news articles.

# Feature Generation
In the feature_gen folder, specify which features you want to generate in the feature_config.yaml file.
Then, run generate_features.py.

# Pipeline
The pipeline contains two sets of python code:

- `data_prep.py`: this code takes a set of labeled cases with features and builds out test & train models. It can be used to run cross-validation on a selected model or to run the model loop.
- `model_loop.py`: this code takes the output csvs from data_prep.py and runs them through various classification models. Results of those models performance on test sets is sent to an output csv for analysis. 

We implement these two files differently via shell code depending on end goal. Each implementation will be named with its end goal in mind. 
