Buttle
======

Buttle is a collection of programs built around nltk with the eventual goal of acceptable word sense
disambiguation.

Notes on PyPy
=============

PyPy is a fast python interpreter that is mostly compatible with cpython, the default interpreter.
Buttle is written to be usable on both PyPy and cpython, and so contains a hack to fix a problem
with PyPy's numpy/pickle implementation.  See classify.py for more information.

The instructions given in this file are tailored to using PyPy.  Adapt them to the regular python
interpreter if desired, but note that training times will be significantly longer.

Installation
============

Create a virtual environment for python:

	$ virtualenv --distribute -p /usr/bin/pypy venv

Activate the virtual environment:

	$ source venv/bin/activate

Install the required packages using pip:

	$ pip install -r requirements.txt

If you are planning on training a maximum entropy model, download megam from:

	http://www.umiacs.umd.edu/~hal/megam/

Place the megam binary in the project directory, and update the path in trainer.py

Temporary Files
===============

Buttle creates a series of temporary files used to speed up the training process.  The most notable
cached objects are the post-processed corpus files in .pickle.  These files are created after post-
processing a corpus.  In most cases, it is significantly faster to load the pickled corpus compared
to the unpickled one.

Buttle's trainer will output pickled taggers after completing, the name will be based on the options
selected for training.
