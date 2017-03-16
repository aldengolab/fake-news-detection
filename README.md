# fake-news-detection
Final Project for CAPP 30255 Advanced Machine Learning for Public Policy  
Lauren Dyson & Alden Golab, Winter 2017  

You can find the final summary for this project in the main directory.

## Dependencies
- python 3.x
- nltk
- numpy
- sklearn
- scipy
- pandas
- spacy

We have included a conda environment yaml. To install, run: 

        conda env create -f environment.yml

## Data

This project is using a [dataset published by Signal Media](http://research.signalmedia.co/newsir16/signal-dataset.html) in conjunction with the Recent Trends in News Information Retrieval 2016 conference to facilitate conducting research on news articles. We use [OpenSources.co](http://opensources.co) to distinguish between 'legitimate' and 'fake' news sources. 

## Feature Generation

From the raw article text, we generate the following features:

1. Vectorized bigram Term Frequency-Inverse Document Frequency, with preprocessing to strip out named entities (people, places etc.) and replace them with anonymous placeholders (e.g. "Donald Trump" --> "-PERSON-"). We use Spacy for tokenization and entity recognition, and SkLearn for TFIDF vectorization.
2. Normalized frequency of parsed syntacical dependencies. Again, we use Spacy for parsing and SkLearn for vectorization. Here is an [excellent interactive visualization](https://demos.explosion.ai/displacy/) of Spacy's dependency parser.

## Pipeline

The pipeline contains four sets of python code:

- `model.py`: a class for models
- `model_loop.py`: a class for running a loop of classifiers; takes test-train data splits and various run params
- `run.py`: this code implements the model_loop class; it also implements our re-sampling of the data
- `transform_features.py`: this file executes all feature generation

Running the code is done through `run.py` with the following options:

1. `--models`: models to run
2. `--iterations`: number of tuning parameter iterations to run per model
3. `--output_dir`: a directory name to store the output; system will create the directory
4. `--dedupe`: whether to look for and remove duplicate content
5. `--features`: which feature set to run with, options include:  

    - `both_only`: runs both PCFG and TFIDF
    - `grammar_only`: runs only PCFG
    - `tfidf_only`: runs only TFIDF
    - `all`: runs all the above

## Example Pipeline Run

To execute the pipeline with Logistic Regression and Stochastic Gradient Descent, navigate to the pipeline directory, run:

```
source activate amlpp
python run.py /path/to/data --models LR SGD --iterations 50 --output_dir run_name --dedupe --reduce 500 --features both_only
```
This is encapsulated in the `run.sh` file. 

The first argument is the path to the input datafile. The pipeline assumes that the text of each article is unique. If your texts are not unique, use the flag --dedupe to automatically remove duplicated articles during preprocessing. To see a description of all arguments, run:

```
python run.py --h
```

A simple report on the models run with basic evaluation metrics will be output to the output/ directory (unless another output directory is specified).


## License

MIT License

Copyright (c) 2017

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
