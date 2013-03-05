from corpus_reader import read_corpus
from nltk import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag.sequential import ClassifierBasedPOSTagger
from classify import megam, MaxentClassifier

from nltk.tag.brill import SymmetricProximateTokensTemplate,        \
                            ProximateTagsRule, ProximateWordsRule,  \
                            ProximateTokensTemplate,                \
                            FastBrillTaggerTrainer
try:
    import cPickle as pickle
except ImportError:
    import pickle

def process_semcor(data):
    b = []
    for sent in data:
        s = []
        for chunk in sent:
            for word in chunk:
                s.append((word, chunk.node))
        b.append(s)
    return b

def train_unigram(train_data, backoff=None, cutoff=0):
    return UnigramTagger(train_data, backoff=backoff, cutoff=cutoff)

def train_bigram(train_data, backoff=None, cutoff=0):
    return BigramTagger(train_data, backoff=backoff, cutoff=cutoff)

def train_trigram(train_data, backoff=None, cutoff=0):
    return TrigramTagger(train_data, backoff=backoff, cutoff=cutoff)

def train_naive(train_data, backoff=None):
    return ClassifierBasedPOSTagger(train=train_data, backoff=backoff)

def _maxent_train(max_iter, min_lldelta):
    return lambda train_feats: MaxentClassifier.train(train_feats, algorithm='megam', max_iter=max_iter, min_lldelta=min_lldelta)

def train_maxent(train_data, max_iter, min_lldelta, backoff=None):
    megam.config_megam('/home/hellfire/Code/buttle/megam_i686.opt')
    return ClassifierBasedPOSTagger(backoff=backoff, train=train_data,
                                    classifier_builder=_maxent_train(max_iter,
                                                                     min_lldelta))

def train_brill(train_data, n_rules, initial=None, backoff=None):
    templates = [
        SymmetricProximateTokensTemplate(ProximateTagsRule, (1,1)),
        SymmetricProximateTokensTemplate(ProximateTagsRule, (2,2)),
        SymmetricProximateTokensTemplate(ProximateTagsRule, (1,2)),
        SymmetricProximateTokensTemplate(ProximateTagsRule, (1,3)),
        SymmetricProximateTokensTemplate(ProximateWordsRule, (1,1)),
        SymmetricProximateTokensTemplate(ProximateWordsRule, (2,2)),
        SymmetricProximateTokensTemplate(ProximateWordsRule, (1,2)),
        SymmetricProximateTokensTemplate(ProximateWordsRule, (1,3)),
        ProximateTokensTemplate(ProximateTagsRule, (-1, -1), (1,1)),
        ProximateTokensTemplate(ProximateWordsRule, (-1, -1), (1,1)),
    ]

    trainer = FastBrillTaggerTrainer(initial_tagger=initial,
                                     templates=templates, trace=3,
                                     deterministic=True)

    return trainer.train(train_data, max_rules=n_rules)

def _tagger_name(tagger):
    '''Returns the name of a single tagger'''
    name = type(tagger).__name__.replace('Tagger', '').lower()
    if name == 'classifierbasedpos':
        name = type(tagger._classifier).__name__.replace('Classifier', '').lower()
    return name


def _tagger_chain(tagger):
    '''Returns a hyphen seperated string of tagger names'''
    stack = [_tagger_name(tagger)]
    current = getattr(tagger, 'backoff', getattr(tagger, '_initial_tagger', None))
    while current:
        name = _tagger_name(current)
        stack.append(name)
        current = getattr(current, 'backoff', getattr(current, '_initial_tagger', None))
    stack.reverse()
    return '-'.join(stack)

def write_tagger(tagger):
    with open(_tagger_chain(tagger) + '.pickle', 'wb') as f:
        pickle.dump(tagger, f)

def _populate_args(parser):
    parser.add_argument('tagger', help='The name of the tagger to train')
    parser.add_argument('-n', '--number', help='The number of sentences to train with', type=int)
    parser.add_argument('-t', '--test', help='The number of sentences to test with', type=int)
    parser.add_argument('-i', '--iterations', help='Max iterations to run', type=int)
    parser.add_argument('-d', '--minlldelta', help='Min ll delta for maxent', type=float)
    parser.add_argument('-b', '--backoff', help='Specify the name of a pickled tagger to use as the backoff tagger')
    parser.add_argument('-c', '--cutoff', help='Any n-gram with less than cutoff occurances will be discarded', type=int)
    parser.add_argument('-r', '--rules', help='Number of rules for a brill tagger', type=int)
    parser.add_argument('-a', '--initial', help='The name of a pickled tagger to use as the initial tagger')

if __name__ == '__main__':
    # Parse the command line options
    from argparse import ArgumentParser
    arg_parser = ArgumentParser()
    _populate_args(arg_parser)
    args = arg_parser.parse_args()

    # Read in the processed corpus
    corpus = read_corpus('semcor', process_semcor)

    # Figure out how many test and train sentences to use
    n_test = args.test
    if n_test is None:
        n_test = int(len(corpus) * 0.003)

    n_train = args.number
    if n_train is None:
        n_train = len(corpus) - n_test

    # Set default values for optional arguments
    n_iterations = args.iterations
    if n_iterations is None:
        n_iterations = 15

    min_lldelta = args.minlldelta
    if min_lldelta is None:
        min_lldelta = 0.01

    cutoff = args.cutoff
    if cutoff is None:
        cutoff = 0

    rules = args.rules
    if rules is None:
        rules = 100

    # Cut up the corpus
    test_data = corpus[:n_test]
    train_data = corpus[n_test:n_test+n_train]

    # Load the backoff tagger if specified
    if args.backoff:
        with open(args.backoff + '.pickle', 'rb') as f:
            backoff = pickle.load(f)
    else:
        backoff = None

    # Load the initial tagger if specified
    if args.initial:
        with open(args.initial + '.pickle', 'rb') as f:
            initial = pickle.load(f)
    else:
        initial = None

    # Run the training!
    if args.tagger == 'unigram':
        tagger = train_unigram(train_data, backoff=backoff, cutoff=cutoff)
    elif args.tagger == 'bigram':
        tagger = train_bigram(train_data, backoff=backoff, cutoff=cutoff)
    elif args.tagger == 'trigram':
        tagger = train_trigram(train_data, backoff=backoff, cutoff=cutoff)
    elif args.tagger == 'naivebayes':
        tagger = train_naive(train_data, backoff=backoff)
    elif args.tagger == 'maxent':
        tagger = train_maxent(train_data, n_iterations, min_lldelta, backoff=backoff)
    elif args.tagger == 'brill':
        tagger = train_brill(train_data, rules, initial=initial, backoff=backoff)

    # Evaluate the tagger
    print args.tagger + ':', tagger.evaluate(test_data)

    # Save the tagger to a file
    write_tagger(tagger)
