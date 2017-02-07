# fake-news-detection
Final Project for CAPP 30255 Advanced Machine Learning for Public Policy  
Lauren Dyson & Alden Golab, Winter 2017  
_This project is currently in progress_.

## Dependencies
- python 3.x
- nltk
- numpy
- sklearn
- scipy
- pandas

## Data

This project is using a [dataset published by Signal Media](http://research.signalmedia.co/newsir16/signal-dataset.html) in conjunction with the Recent Trends in News Information Retrieval 2016 conference to facilitate conducting research on news articles. We use [OpenSources.co](http://opensources.co) to distinguish between 'legitimate' and 'fake' news sources. 

## Feature Generation

In the feature_gen folder, specify which features you want to generate in the feature_config.yaml file.
Then, run `generate_features.py`.

## Pipeline

The pipeline contains two sets of python code:

- `test_train.py`: this code takes a set of labeled cases with features and builds out test & train models. It can be used to run cross-validation on a selected model or to run the model loop.
- `model_loop.py`: this code takes the output csvs from data_prep.py and runs them through various classification models. Results of those models performance on test sets is sent to an output csv for analysis. 

We implement these two files differently via shell code depending on end goal. Each implementation will be named with its end goal in mind. 

## License

MIT License

Copyright (c) 2017

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
