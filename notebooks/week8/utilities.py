import copy
import math
import random
import numpy as np



from statistics import mean, stdev
from collections import defaultdict



def Entropy(prob_X):
    """
    This is the entropy function that computes
    entropy  of a  given  random  variables  X
    and with  their corresponding probabilities
    p_i based on the definition in:

    Shanon and Weaver, 1949

    -> Links to paper :
    --> http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
    --> https://ieeexplore.ieee.org/document/6773024


              Entropy = Σ_i p_i * log2 (p_i)


    INPUT:
    -------
            prob_X (a list/array of vairables):

       it should contains all the  probabilities of
       the underlying random variable, each element
       expected to be a (0 <= float) and should
       add up to 1. (Else will be normalized)



    OUTPUT:
    -------
            Entropy (float): Entropy bits
    """
    import math
    _sum_ = 0

    _tot_ = 0
    # checks
    for prob in prob_X:
        assert prob >= 0, "Negetive probability is not accepted!!!"
        _tot_ += prob

#     if _tot_!=1:
#         print("Inputs are not normalized added up to {}, will be normalized!!".format(_tot_))

    for prob in prob_X:
        if _tot_==0:
            continue

        prob = prob/_tot_
        if prob == 0:
            pass
        else:
            _sum_ += prob * math.log2(prob)

    return abs(_sum_)


def Boolean_Entropy(q):
    """
    Finds the entropy for a Boolean random variable.

    INPUT:
    ------
           q (float) : is expected to be between 0 and 1 (else AssertionError)

    OUTPUT:
    -------
            Entropy (float) : Entropy of a throwing a coin with chances
                              of P(H, T) = (q, 1 - q) in bits


    """
    assert q >= 0 and q <= 1, "q = {} is not between [0,1]!".format(q)

    return Entropy([q, 1-q])


def Boolean_Entropy_counts(p, n):
    """
    Finds the entropy for a Boolean random variable.

    INPUT:
    ------
           p (int or float) : Number or relative fraction of positive instances
           n (int or float) : Number or relative fraction of negative instances

    OUTPUT:
    -------
            Entropy (float) : Entropy of a throwing a coin with chances
                              of P(H, T) = (q, 1 - q) in bits

                              with q = p / (n + p)

    """
    if n==0 and p==0:
        return 0
    q = p / (n + p)
    return Boolean_Entropy(q)




def Remainder_Entropy(Attr, outcome):

    set_of_distinct_values = set(Attr)

    count_distict_values = len(set_of_distinct_values)
    count_distict_outcomes = len(set(outcome))

    assert count_distict_outcomes <= 2, "{} different outcomes but expected Boolean"


    count_total_positives = len([i for i in outcome if i!=0])
    count_total_negatives = len(outcome) - count_total_positives

    import numpy as np

    Attr_np = np.array(Attr)
    outcome_np = np.array(outcome)

    _sum_ = 0

    for value in set_of_distinct_values:
        _outcome_ = outcome_np[Attr_np==value]
        count_positives = len([i for i in _outcome_ if i!=0])
        count_negatives = len(_outcome_) - count_positives

        _entropy_ = Boolean_Entropy_counts(count_positives, count_negatives)
        _weights_ = (count_positives + count_negatives)
        _weights_ = _weights_ / (count_total_positives + count_total_negatives)

        _sum_ += _weights_ * _entropy_

    return _sum_



def Information_Gain(Attr, outcome):
    count_total_positives = len([i for i in outcome if i!=0])
    count_total_negatives = len(outcome) - count_total_positives

    intital_entropy = Boolean_Entropy_counts(count_total_positives, count_total_negatives)
    remaining_entropy = Remainder_Entropy(Attr, outcome)

    info_gain = intital_entropy - remaining_entropy

    return info_gain



def euclidean_distance(X, Y):
    return math.sqrt(sum((x - y)**2 for x, y in zip(X, Y)))


def cross_entropy_loss(X, Y):
    n=len(X)
    return (-1.0/n)*sum(x*math.log(y) + (1-x)*math.log(1-y) for x, y in zip(X, Y))


def rms_error(X, Y):
    return math.sqrt(ms_error(X, Y))


def ms_error(X, Y):
    return mean((x - y)**2 for x, y in zip(X, Y))


def mean_error(X, Y):
    return mean(abs(x - y) for x, y in zip(X, Y))


def manhattan_distance(X, Y):
    return sum(abs(x - y) for x, y in zip(X, Y))


def mean_boolean_error(X, Y):
    return mean(int(x != y) for x, y in zip(X, Y))


def hamming_distance(X, Y):
    return sum(x != y for x, y in zip(X, Y))


def _read_data_set(data_file, skiprows=0, separator=None):
    with open(data_file, "r") as f:
        file = f.read()
        lines = file.splitlines()
        lines = lines[skiprows:]

    data_ = [[] for _ in range(len(lines))]

    for i, line in enumerate(lines):
        splitted_line = line.split(separator)
        float_line = []
        for value in splitted_line:
            try:
                value = float(value)
            except ValueError:
                if value=="":
                    continue
                else:
                    pass
            float_line.append(value)
        if float_line:
            data_[i] = float_line

    for line in data_:
        if not line:
            data_.remove(line)

    return data_

def unique(seq):
    """
    Remove any duplicate elements from any sequence,
    works on hashable elements such as int, float,
    string, and tuple.
    """
    return list(set(seq))


def remove_all(item, seq):
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]


def weighted_sample_with_replacement(n, seq, weights):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight."""
    sample = weighted_sampler(seq, weights)

    return [sample() for _ in range(n)]


def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    import bisect

    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def mode(data):
    import collections
    """Return the most common data item. If there are ties, return any one of them."""
    [(item, count)] = collections.Counter(data).most_common(1)
    return item

# argmin and argmax

identity = lambda x: x

argmin = min
argmax = max


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return argmin(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return argmax(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items

def check_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)



def Measure_accuracy(true_values, predictions):
    _sum_ = 0
    for truth, prediction in zip(true_values, predictions):
        if truth==prediction:
            _sum_+=1
    return _sum_/len(predictions)

# Kernels for the SVM
def gaussian_kernel(X1, X2, sigma):
    '''
    INPUT:
    -------
            X1 : shape (N examples, n features)
            X2 : shape (N examples, n features)
            sigma : Parameter for gaussian kernel (rbf)
    OUTPUT:
    -------
            kernel: shape (N examples, N examples)
    '''
    return np.exp((-(np.linalg.norm(X1[None, :, :] - X2[:, None, :], axis=2) ** 2)) / (2 * sigma ** 2))

def linear_kernel(X1, X2, *args):
    '''
    INPUT:
    -------
            X1 : shape (N examples, n features)
            X2 : shape (N examples, n features)

    OUTPUT:
    -------
            kernel: shape (N examples, N examples)
    '''
    return np.tensordot(X2, X1, axes=(1, 1))
