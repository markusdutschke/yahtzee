#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:24:16 2019

@author: user

Some statement about module
"""
# --- imports
#import inspect
from inspect import getframeinfo, currentframe
import time

def lineno():
    """Returns the current line number in our program."""
    return currentframe().f_back.f_lineno


def getFrameLine():
    #works
    frameinfo = getframeinfo(currentframe().f_back.f_back)
    fn,ln=frameinfo.filename, frameinfo.lineno
    return fn,ln
    #only linenumber
    return currentframe().f_back.f_back.f_lineno

def lp(*args, **kwargs): #print with additionally filename and line number
    fn,ln=getFrameLine()
    txt='LINE NUMBER {:d} - {:}, {:}\n\t'.format(
            ln,
            fn.split('/')[-1],
            str(time.ctime())
            )
    sep=kwargs.get('sep')
    if sep is None:
        sep=' '
    mergedArgs=sep.join([str(arg) for arg in args])
    txt= txt+mergedArgs.replace('\n','\n\t')
    print(txt, **kwargs)


# doctest example: https://stackoverflow.com/a/23188939/7128154
def func_docstring_docu(arg1, arg2):
    """Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    See Also
    --------
    otherfunc : some related other function

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a=[1,2,3]
    >>> [x + 3 for x in a]
    [4, 5, 6]
    """
    return

