#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:29:47 2019

@author: user
"""

import numpy as np
from comfct.debug import lp


def weighted_choice(items, weights):
    assert len(items) == len(weights)
    wSum = np.sum(weights)
    ws = [w/wSum for w in weights]
    
    #works for items being list of lsits
    inds = list(range(len(items)))
    ind = np.random.choice(inds, p=ws)
    
    return items[ind]
    
#    return np.random.choice(items, p=ws)


#https://stackoverflow.com/a/23979509
def arreq_in_list(myarr, list_arrays):
    """Check if array is in list of arrays"""
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)
