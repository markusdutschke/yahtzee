#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:10:41 2019

@author: user
"""
import sys

# https://stackoverflow.com/a/34482761
def progressbar(it, prefix="", size=50):
    count = len(it)
    def show(j):
        x = int(size*j/count)
#        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
#        file.flush()
        print("\r%s [%s%s] %i/%i"%(prefix, '\u2588'*x, "."*(size-x), j, count),
              end='', flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print()
#    file.write("\n")
#    file.flush()