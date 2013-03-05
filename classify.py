from nltk.classify import *
from copy import copy

_NltkMaxentClassifier = MaxentClassifier

class MaxentClassifier(object):
    '''Proxies nltk's MaxentClassifier to provide pickling in PyPy'''
    def __getstate__(self):
        classifier = copy(self._classifier)
        classifier._weights = classifier._weights.tostring()
        return classifier

    def __setstate__(self, classifier):
        classifier._weights = numpy.fromstring(classifier._weights)
        self._classifier = classifier

    def classify(self, *args, **kwargs):
        return self._classifier.classify(*args, **kwargs)

    @classmethod
    def train(cls, *args, **kwargs):
        c = MaxentClassifier()
        c._classifier = _NltkMaxentClassifier.train(*args, **kwargs)
        return c
