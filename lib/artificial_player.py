#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:02:57 2019

@author: user
"""
from comfct.debug import lp
from abc import ABC, abstractmethod
import numpy as np
from yahtzee import Game

def benchmark(player, nGames=100):
    """Benchmarks Yahtzee decision making models.
    
    Extended description of function.
    
    Parameters
    ----------
    fctRoll : function
        Classifier for which dice to reroll.
        arg1 : ScoreBoard
        arg2 : Dice
        arg3 : int, 0 or 1
            number of reroll attempts so far
        returns : bool array of len 5
    fctCat : str
        Classifier for which categorie to use on the score board for dice.
        arg1 : ScoreBoard
        arg2 : Dice
        returns : 0 <= int <= 12
    player : Player
        Artificial player, subclass of AbstractPlayer
    nGames : int
        number of trials
    
    Returns
    -------
    scoreMean : float
        Description of return value
    scoreStd : float
        Standard deviation
    
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
    scores = []
    for ii in range(nGames):
        game = Game(player)
        scores += [game.sb.getSum()]
    return np.mean(scores), np.std(scores)
    

class AbstractPlayer(ABC):  # abstract class
 
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @abstractmethod
    def fct_roll(self, scoreBoard, dice, attempt):
        """Decides what dice to reroll.
        
        Extended description of function.
        
        Parameters
        ----------
        scoreBoard : ScoreBoard
            The score board
        dice : Dice
            Current dice configuration
        attempt : 0 <= int <=1
            How many attempts of throwing dice so far in this round
        
        Returns
        -------
        scoreMean : float
            Description of return value
        scoreStd : float
            Standard deviation
        
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
        pass
    
    @abstractmethod
    def fct_cat(self, scoreBoard, dice):
        """Decides what category to choose.
        
        Extended description of function.
        
        Parameters
        ----------
        scoreBoard : ScoreBoard
            The score board
        dice : Dice
            Current dice configuration
        
        Returns
        -------
        cat : int
            index of the category
            Can not return a category, which was used already
        
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
        pass
    
class PlayerRandomCrap(AbstractPlayer):
    """This player behaves completely random"""
    name = 'Random Crap'
    def fct_roll(self, scoreBoard, dice, attempt):
        return np.random.choice([True, False], 5)
    def fct_cat(self, scoreBoard, dice):
        return np.random.choice(scoreBoard.open_cats())

class PlayerOneShotHero(AbstractPlayer):
    """This player assigns the dice to the category with the most scores"""
    name = 'Mr. One Shot Hero'
    def fct_roll(self, scoreBoard, dice, attempt):
        return [False]*5
    def fct_cat(self, scoreBoard, dice):
#        print(151, dice, type(dice))
        bench = []
        for cat in scoreBoard.open_cats():
            bench += [(scoreBoard.check_points(dice, cat), cat)]
#        lp(bench)
        bench = sorted(bench, key=lambda x: x[0])
        return bench[-1][1]

class PlayerOneShotAI(AbstractPlayer):
    """No strategic dice reroll, but self learning category assignment"""
    name = 'The One Shot AI'
    def fct_roll(self, scoreBoard, dice, attempt):
        return [False]*5
    def fct_cat(self, scoreBoard, dice):  # TODO
        bench = []
        for cat in scoreBoard.open_cats():
            bench += [(scoreBoard.check_points(dice, cat), cat)]
        bench = sorted(bench, key=lambda x: x[0])
        return bench[-1][1]
    def train(self, nGames):
        """use:
            MLPRegressor
            partial_fit
            https://www.programcreek.com/python/example/93778/sklearn.neural_network.MLPRegressor
        """
        