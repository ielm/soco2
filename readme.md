Social Computing Homework 2
===============================

soco2 is a cli wrapper for training and evaluating a gender classifier on twitter data for Social Computing with Dr. Tomek Strzalkowski. 

Installing GloVe 
----------------
Download and unzip the [GloVe pretrained word vectors](http://nlp.stanford.edu/data/glove.6B.zip) in `/soco2/model/gender/`


Usage
-----
# For usage instructions, run
```shell script
$ python soco.py --help
```
# To build the data directories

```shell script
$ python soco.py build
```

# To train the model
```shell script
$ python soco.py train
```

# To evaluate the model
```shell script
$ python soco.py evaluate
```