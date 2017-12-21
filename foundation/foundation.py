# -*- coding: utf-8 -*-

for i in range(0,100)
    print i

def add_to_dict(args={'a': 1, 'b': 2}):
    for i in args.keys():
        args[i] += 1
    print args

def fibo(n):
    a = 1
    b = 1
    c = 1
    arr = []
    for i in range(0,n):
        d = a
        a = a + b
        b = d
        c += 1
        arr.append(a)
    print a , b, arr

def quickSort(arr,low,high):
    i = low
    j = high
    if i >= j:
        return arr
    key = arr[i]

    while i < j:
        while i < j and arr[j] >= key:
            j -= 1
        arr[i] = arr[j]
        while i < j and arr[i] <= key:
            i += 1
        arr[j] = arr[i]
    arr[i] = key
    quickSort(arr,low,i - 1)
    quickSort(arr,j + 1,high)
    return arr

import numpy as np

def primeList(n):
    is_prime = np.ones((n,),dtype=bool)
    is_prime[:2] = 0

    nMax = int(np.sqrt(len(is_prime)))
    for j in range(2,nMax):
        is_prime[ 2*j : : j ] = False
