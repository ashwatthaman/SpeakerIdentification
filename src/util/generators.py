#coding:utf-8
import codecs
def batch(generator, batch_size):
    batch = []
    is_tuple = False
    for l in generator:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch

def sorted_parallel(generator1, generator2, pooling, order=1):
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        for x in zip(batch1, batch2):
            yield x

def word_list(filename):
    with codecs.open(filename,encoding="utf-8") as fp:
        for l in fp:
            yield l.split()

def class_list(filename):
    with codecs.open(filename,encoding="utf-8") as fp:
        for l in fp:
            yield int(l)

def letter_list(filename):
    with open(filename) as fp:
        for l in fp:
            yield list(''.join(l.split()))

