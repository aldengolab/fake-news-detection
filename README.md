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
- spacy

## Data

This project is using a [dataset published by Signal Media](http://research.signalmedia.co/newsir16/signal-dataset.html) in conjunction with the Recent Trends in News Information Retrieval 2016 conference to facilitate conducting research on news articles. We use [OpenSources.co](http://opensources.co) to distinguish between 'legitimate' and 'fake' news sources. 

## Feature Generation

From the raw article text, we generate the following features:

1. Vectorized bigram Term Frequency-Inverse Document Frequency, with preprocessing to strip out named entities (people, places etc.) and replace them with anonymous placeholders (e.g. "Donald Trump" --> "-PERSON-"). We use Spacy for tokenization and entity recognition, and SkLearn for TFIDF vectorization.
2. Normalized frequency of parsed syntacical dependencies. Again, we use Spacy for parsing and SkLearn for vectorization. Here is an [excellent interactive visualization](https://demos.explosion.ai/displacy/) of Spacy's dependency parser.

## Pipeline

The pipeline contains two sets of python code:

- `test_train.py`: this code takes a set of labeled cases with features and builds out test & train models. It can be used to run cross-validation on a selected model or to run the model loop.
- `model_loop.py`: this code takes the output csvs from data_prep.py and runs them through various classification models. Results of those models performance on test sets is sent to an output csv for analysis. 

We implement these two files differently via shell code depending on end goal. Each implementation will be named with its end goal in mind. 

## Example Pipeline Run

To execute the python with Logistic Regression and Naive Bayes, navigate to the pipeline directory, run:
```
python run.py ../articles_deduped.csv --models LR NB
```

The first argument is the path to the input datafile. To see a description of all arguments, run:

```
python run.py --h
```

## License

MIT License

Copyright (c) 2017

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
