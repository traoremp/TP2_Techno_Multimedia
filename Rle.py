#https://rosettacode.org/wiki/Zig-zag_matrix#Python
import numpy as np
import time 
def sortComp(pair):
    x = pair[0]
    y = pair[1]
    return (x+y, -y if (x+y) % 2 else y)

def getZigZag(squareMat):
    n = len(squareMat)
    array = []
    indexorder = sorted(((x, y) for x in range(n) for y in range(n)),
                        key=sortComp)
    for (x,y) in indexorder:
        array.append(squareMat[x][y])
    return array

from heapq import heappush, heappop, heapify
from collections import defaultdict
 
def encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
def huffman(zigZagArray):
    symb2freq = defaultdict(int)
    for el in zigZagArray[1:]:
        symb2freq[el] += 1
    # in Python 3.1+:
    # symb2freq = collections.Counter(txt)
    huff = encode(symb2freq)
    return huff
    # print "Symbol\tWeight\tHuffman Code"
    # for p in huff:
    #     print "%s\t%s\t%s" % (p[0], symb2freq[p[0]], p[1])
def DPCM(DC_Values):
    pass

def rle(zigZagArray, huffman_symbole_codes):
    number_of_preceding_zeros = 0
    for data in zigZagArray[1:]:
        if data == 0:
            number_of_preceding_zeros += 1
            if number_of_preceding_zeros > 15:
                pass
        else:
            pass




