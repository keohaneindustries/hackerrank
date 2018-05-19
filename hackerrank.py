#!/bin/python3

import math
import os
import random
import re
import sys

from itertools import product

def maximize(m, nl, el):
    f = 1
    
    for n in nl:
        f=f*n
    
    L = []
    for i in range(f):
        L.append([])
        
    for i in range(len(nl)-1):
        for j, x in enumerate(el[i]):
            for w in range(f/nl[i]):
                L[w*j*(1+i)].append(x**2)

    maxed = []
    maxv = 0

    curr = []
    for l in L:
        if (sum(l) % m) > maxv:
            maxv = sum(l) % m
            maxed = l.copy(deep=True)
    
    return maxed


if __name__ == '__main__':
    km = input().split()
    
    k = int(km[0])
    
    m = int(km[1])
    
    nl = []
    el = []
    
    for _ in range(k):
        matrix_item = map(int, input().split())
        nl.append(matrix_item.__next__())
        el.append(list(matrix_item))
    
    print(maximize(m, nl, el))

