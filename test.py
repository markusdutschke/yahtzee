#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:41:30 2019

@author: user
"""

import numpy as np

def split_in_consec_ints(lst):
    splits = [0]
    for ii, el in enumerate(lst[:-1]):
        if el + 1 < lst[ii+1]:
            splits += [ii+1]
    splits += [len(lst)]
    print(splits)
    slst = [lst[a:b] for a,b in zip(splits, splits[1:])]
    return slst

print(split_in_consec_ints([1,3,4, 5 , 7,8, 10]))
print(sum([1, 2], 0))
