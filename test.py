#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:41:30 2019

@author: user
"""

import numpy as np

a= np.array([ [1,2], [4,3], [6,1], [4,2]])
print(a)
#print(np.sort(a, axis=0))
print( sorted(a, key=lambda x: x[0]))

