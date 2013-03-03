import code
from nltk.corpus import wordnet

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
