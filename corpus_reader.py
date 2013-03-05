import nltk.corpus
from hashlib import sha1
import os
from bz2 import BZ2File

try:
    import cPickle as pickle
except ImportError:
    import pickle

'''Saves a small amount of time when importing a corpus'''

def _funcsig(func):
    '''Creates a unique signature for a python function'''
    if not func:
        return 'default'
    bytecode = func.func_code.co_code
    return sha1(bytecode).hexdigest()

PICKLED_DIRECTORY = '.pickled'

def _write_corpus(name, postfunc, pickle_file, args, kwargs):
    '''Writes a compressed pickled corpus to a file'''
    with BZ2File(pickle_file, 'w', 0, 1) as f:
        corpus = getattr(nltk.corpus, name)

        if not args:
            args = []
        if not kwargs:
            kwargs = {}

        # Read the tagged sentences from the corpus
        sents = list(corpus.tagged_sents(*args, **kwargs))

        # Apply the post processing function
        if postfunc:
            sents = postfunc(sents)

        # Write the processed corpus out
        pickle.dump(sents, f)
        return sents

def _read_corpus(pickle_file):
    with BZ2File(pickle_file, 'r') as f:
        return pickle.load(f)

def read_corpus(name, postfunc=None, args=None, kwargs=None):
    '''Returns the named corpus after applying the post processing function'''
    # Create directory for corpus pickles
    if os.path.exists(PICKLED_DIRECTORY):
        if not os.path.isdir(PICKLED_DIRECTORY):
            raise Exception('pickled corpus destination exists and is not a directory')
    else:
        os.mkdir('.pickled')

    pickle_file = PICKLED_DIRECTORY + '/{}-{}.pickle.bz2'.format(name, _funcsig(postfunc))

    if not os.path.isfile(pickle_file) or os.stat(pickle_file).st_size == 0:
        return _write_corpus(name, postfunc, pickle_file, args, kwargs)
    else:
        return _read_corpus(pickle_file)
