#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:07:52 2019

@author: user
"""

def list_cast(x): #casts x to a list
    """if object is a list or itaerable: return a list
    otherwise create a list with x being the only element in it
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]