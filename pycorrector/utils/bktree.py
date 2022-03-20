#from itertools import imap, ifilter
import os
from pycorrector.utils.text_utils import get_unify_pinyins

class BKTree:
    def __init__(self, distfn, words):
        """
        Create a new BK-tree from the given distance function and
        words.

        Arguments:
        distfn: a binary function that returns the distance between
        two words.  Return value is a non-negative integer.  the
        distance function must be a metric space.

        words: an iterable.  produces values that can be passed to
        distfn

        """
        self.distfn = distfn

        it = iter(words)
        root = it.__next__()
        self.tree = (root, {})

        for i in it:
            self._add_word(self.tree, i)

    def _add_word(self, parent, word):
        pword, children = parent
        d = self.distfn(word, pword)
        if d in children:
            self._add_word(children[d], word)
        else:
            children[d] = (word, {})

    def query(self, word, n):
        """
        Return all words in the tree that are within a distance of `n'
        from `word`.
        Arguments:

        word: a word to query on
        n: a non-negative integer that specifies the allowed distance
        from the query word.

        Return value is a list of tuples (distance, word), sorted in
        ascending order of distance.

        """

        def rec(parent):
            pword, children = parent
            d = self.distfn(word, pword)
            results = []
            if d <= n:
                results.append((d, pword))

            for i in range(d - n, d + n + 1):
                child = children.get(i)
                if child is not None:
                    results.extend(rec(child))
            return results

        # sort by distance
        return sorted(rec(self.tree))

# def maxdepth(tree, count=0):
#     _, children = t
#     if len(children):
#         return max(maxdepth(i, c + 1) for i in children.values())
#     else:
#         return c


# http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Levenshtein_distance#Python
def levenshtein(s, t):
    m, n = len(s), len(t)
    d = [range(n + 1)]
    d += [[i] for i in range(1, m + 1)]
    for i in range(0, m):
        for j in range(0, n):
            cost = 1
            if s[i] == t[j]: cost = 0

            d[i + 1].append(min(d[i][j + 1] + 1,  # deletion
                                d[i + 1][j] + 1,  # insertion
                                d[i][j] + cost)  # substitution
                            )
    return d[m][n]


def timeof(fn, *args):
    import time
    t = time.time()
    res = fn(*args)
    print ("time: ", (time.time() - t))
    return res

if __name__ == "__main__":
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    freq_words_pinyins = []
    with open(pwd_path+'/../data/word_freq.txt', 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            line2 = line.strip()
            if not line2 or len(line2) == 0:
                continue
            parts = line2.split(' ')
            pinyins = get_unify_pinyins(parts[0].lower())
            freq_words_pinyins.append(''.join(pinyins))
            count += 1
            if count >= 1500:
                break
    tree = BKTree(levenshtein, set(freq_words_pinyins))
    print (tree.query("yiyifen", 2))
