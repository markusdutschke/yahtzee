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
#    ws[-1] = 1- np.sum(ws[:-1])
#    lp(items, ws)
    return np.random.choice(items, p=ws)
