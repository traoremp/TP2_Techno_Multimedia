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
    for el in zigZagArray:
        symb2freq[el] += 1
    # in Python 3.1+:
    # symb2freq = collections.Counter(txt)
    huff = encode(symb2freq)
    return huff
    # print "Symbol\tWeight\tHuffman Code"
    # for p in huff:
    #     print "%s\t%s\t%s" % (p[0], symb2freq[p[0]], p[1])
def DPCM(DC_Values):
    predictor = 1.0
    encoding = []
    encoding.append(DC_Values[0])
    for x in range(1, len(DC_Values)):
        encoding.append(DC_Values[x]-(predictor*DC_Values[x-1]))
    return encoding

def rle(zigZagArrays, huffman_symbole_codes):
    number_of_preceding_zeros = 0
    encodings = []
    for x in range(0, len(zigZagArrays)):
        zigzagArr = zigZagArrays[x]
        #AssociatedhuffmanSymCodes = huffman_symbole_codes[x]
        #print zigzagArr, AssociatedhuffmanSymCodes
        for data in zigzagArr:
            if data == 0:
                number_of_preceding_zeros += 1
                if number_of_preceding_zeros > 15:
                    special_entry = (15,0, 0)
                    encodings.append(special_entry)
                    number_of_preceding_zeros = 0
            else:
                #print AssociatedhuffmanSymCodes
                non_zero_encoding = (number_of_preceding_zeros, len(huffman_symbole_codes[data]), data)
                number_of_preceding_zeros = 0
                encodings.append(non_zero_encoding)
        number_of_preceding_zeros = 0
        EOB = (0, 0)
        encodings.append(EOB)
    return encodings
        


def compress(encoded_DC_Values, encoded_image, huffman_symbole_codes):
    final_compression = ''
    #final_compression += '{0:08b}'.format(int(len(encoded_DC_Values)))
    #for value in encoded_DC_Values:
    #    final_compression += '{0:08b}'.format(int(value))
    for an_input in encoded_image:
        if len(an_input) == 2:
            final_compression += '{0:08b}'.format(int(0)) #EOB
        else:
            if an_input[2] == 0:
                final_compression += '{0:08b}'.format(int(0))
            else: 
                final_compression += '{0:04b}'.format(int(an_input[0]))
                final_compression += '{0:04b}'.format(int(an_input[1]))
                final_compression += huffman_symbole_codes[an_input[2]]
    return final_compression