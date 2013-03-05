import code
import nltk
from nltk.corpus import wordnet, semcor
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.stem.lancaster import LancasterStemmer
from corpus_reader import read_corpus

def to_bag(sent):
    bag = list()
    stack = list(sent)

    while stack:
        current = stack.pop()
        if all(isinstance(x, basestring) for x in current):
            pos = current.node
            try:
                sense = current.parent.node
            except AttributeError:
                sense = None

            if not pos or not sense or sense == 'DT':
                continue

            for word in current:
                bag.append((pos, sense, word))
        else:
            children = list(current)
            for x in children:
                x.parent = current
            stack += children

    return bag

def all_to_bag(sents):
    return [to_bag(x) for x in sents]

train_data = read_corpus('semcor', postfunc=all_to_bag, kwargs={'tag': 'both'})
train_data = train_data[:100]
print train_data[0]

def wsd_features(tagged_sent, index):
    features = {}

    target_word, target_pos = tagged_sent[index]
    features['word'] = target_word
    features['pos'] = target_pos

    nouns = []
    verbs = []
    adj = []
    adv = []

    for idx, (word, pos) in enumerate(tagged_sent):
        if idx == index:
            continue
        distance = abs(index-idx)
        if pos[0] == 'N':
            nouns.append((distance, word))
        elif pos[0] == 'V':
            verbs.append((distance, word))
        elif pos[0] == 'J':
            adj.append((distance, word))
        elif pos[0] == 'R':
            adv.append((distance, word))

        for key, value in {'noun': nouns, 'verb': verbs, 'adj': adj, 'adv': adv}.items():
            value.sort()
            for idx, word in enumerate(value[:3]):
                features[key + '-' + str(idx)] = word[1].lower()

    return features

train_set = []
for sent in train_data:
    tagged_sent = [(word[2], word[0]) for word in sent]
    for idx, word in enumerate(sent):
        features = wsd_features(tagged_sent, idx)
        sense = word[1]
        train_set.append((features, sense))

classifier = NaiveBayesClassifier.train(train_set)

code.interact(local=locals())


class Concept(object):
    def __init__(self, *args):
        if args:
            synsets = [wordnet.synsets(x) for x in args]
            self.synsets = self._common_synsets(synsets)
            if len(args) > 1:
                isas = [self._isa_synsets(synsets, x) for x in synsets]
                self.synsets = set.union(self.synsets, *isas)
        else:
            self.synsets = set()

    def _common_synsets(self, synsets):
        sets = list(set(x) for x in synsets)
        return set.intersection(*sets)

    def _has_hypernym(self, target, synsets):
        paths = [set(x) for x in target.hypernym_paths()]
        hypernyms = set.union(*paths)
        return any(x in synsets for x in hypernyms)

    def _isa_synsets(self, synsets, target):
        synsets = [set(x) for x in synsets if x != target]
        synsets = set.intersection(*synsets)
        result = set()

        for synset in target:
            # Check for direct hypernyms
            if self._has_hypernym(synset, synsets):
                result.add(synset)
            else:
                # Check for derivationally related hypernyms
                for lemma in synset.lemmas:
                    if any(self._has_hypernym(x.synset, synsets) for x in lemma.derivationally_related_forms()):
                        result.add(synset)
                        break

                # Check for similar_tos
                for similar in synset.similar_tos():
                    if self._has_hypernym(similar, synsets):
                        result.add(synset)
                        break
        return result

    def define(self):
        return ' *and* '.join(x.definition for x in self.synsets)

    def __add__(self, other):
        new = self.__class__()
        new.synsets = self.synsets | other.synsets
        return new

    def __iadd__(self, other):
        self.synsets |= other.synsets
        return self

    def __sub__(self, other):
        new = self.__class__()
        new.synsets = self.synsets - other.synsets
        return new

    def __isub__(self, other):
        self.synsets -= other.synsets
        return self

    def __and__(self, other):
        new = self.__class__()
        new.synsets = self.synsets & other.synsets
        return new

    def __iand__(self, other):
        self.synsets &= other.synsets
        return self

    def __or__(self, other):
        new = self.__class__()
        new.synsets = self.synsets | other.synsets
        return new

    def __ior__(self, other):
        self.synsets |= other.synsets
        return self

    def __xor__(self, other):
        new = self.__class__()
        new.synsets = self.synsets ^ other.synsets
        return new

    def __ixor__(self, other):
        self.synsets ^= other.synsets
        return self

code.interact(local=locals())
