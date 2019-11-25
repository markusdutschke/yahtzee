#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:53:36 2019

@author: user
"""
# --- imports
import numpy as np
#np.random.seed(0)
import pandas as pd
from comfct.debug import lp
from copy import deepcopy


def roll_dice(nDice=5):
    #dice = np.empty(shape=nDice,dtype=int)
    dice=0
    for ii in range(0,nDice):
        #dice[ii]=random.randint(1,6)
        dice*=10
        dice+=np.random.randint(1,6)
    return dice

class Dice:
    
    def __init__(self, arr=None):
        if arr is None:
            self.vals = np.sort(np.random.randint(1,6,5))
        else:
            assert len(arr) == 5
            self.vals = np.sort(arr)
#    # trivial iteration for lst = list(game) functionality
#    # https://www.programiz.com/python-programming/iterator
#    def __iter__(self):
#        self.isNext = True
#        return self
#    def __next__(self):
#        if self.isNext:
#            self.isNext = False
#            return self
#        else:
#            raise StopIteration
    
    def roll(self, arr):
        """arr is a boolen array"""
        assert len(arr) == 5
#        lp(self.vals)
#        lp(arr)
        newVals = np.random.randint(1, 6, np.sum(arr))
        self.vals[arr] = newVals
        self.vals = np.sort(self.vals)
#        lp(self.vals)
    
    def __str__(self):
        return ', '.join(['{:.0f}'.format(val) for val in self.vals])
    
    def to_str(self, mask = [False]*5):
        """mask : bool array of len 5; marks if dice is intended fore reroll"""
        _str = ','.join(
                [str(v) + 'r' if b else str(v) for v, b in zip(self.vals, mask)])
        return '[{:}]'.format(_str)
#        _str = ''
#        for v, b in zip(self.vals, mask):
#            suf = 'r' if b else ''
#            _str += str(v) + suf
#        return _str
    
    def copy(self):
        return deepcopy(self)
    
    def compress(self):
        """Result of the 5 dice is encoded as a 5 digit integer"""
        pass

class ScoreBoard:
    cats = [
            'Aces','Twos','Threes','Fours','Fives','Sixes',
            'Three Of A Kind','Four Of A Kind','Full House',
            'Small Straight','Large Straight','Yahtzee','Chance']
    def __init__(self):
        self.scores=np.ma.masked_array(np.empty(shape=13),mask=True,dtype=int)
#    # trivial iteration for lst = list(game) functionality
#    # https://www.programiz.com/python-programming/iterator
#    def __iter__(self):
#        self.isNext = True
#        return self
#    def __next__(self):
#        if self.isNext:
#            self.isNext = False
#            return self
#        else:
#            raise StopIteration
    
    @property
    def data(self):
        return self.scores.data
    @property
    def mask(self):
        return self.scores.mask
    
    @property
    def score(self):
        return self.getSum()
    
    def check_points(self, dice, cat):
        """Just check how many points one would get by assigning dice
        to category number cat
        dice:Dice
        cat: int.
        """
#        lp(dice, type(dice))
        dice = dice.vals
#        assert self.scores.mask[cat], (
#                'Mask must be True if dice should be assigned')
        score = 0
        if cat>=0 and cat<=5:
            score=np.sum(dice[dice==(cat+1)])
        elif cat==6:
            if np.amax(np.bincount(dice))>=3:
                score=np.sum(dice)
        elif cat==7:
            if np.amax(np.bincount(dice))>=4:
                score=np.sum(dice)
        elif cat==8:
            sd=np.sort(np.bincount(dice))
            if (sd[-1]==3 and sd[-2]==2) or sd[-1]==5:
                score=25
        elif cat==9:
            sd=np.bincount(dice)
            lenStraight=0
            for ii in range(0,len(sd)):
                if sd[ii]>0:
                    lenStraight+=1
                else:
                    lenStraight=0
                if lenStraight>=4:
                    score=30
                    break
        elif cat==10:
            sd=np.where(np.bincount(dice)>0,1,0)
            if np.sum(sd)==5 and (not (dice==1).any() or not (dice==6).any()):
                score=40
        elif cat==11:
            if np.amax(np.bincount(dice))==5:
                score=50
        elif cat==12:
            score=np.sum(dice)
        else:
            assert False, 'invalid category position, cat='+str(cat)
        return score
    
    
    def add(self, dice, cat):
        """dice: Dice instance; posCat: int index for the category of the sb"""
        #dice: np.ndarray, posCat int
#        assert isinstance(posCat, int)
        assert np.issubdtype(type(cat), np.signedinteger)
        
        assert np.ma.getmask(self.scores)[cat]==1 \
        , str(self.scores)+ '\n'+str(self.open_cats_mask()) \
        +'\nposCat='+str(cat)+'\ntry to add to used category!'
#        dice = dice.vals
        self.scores[cat] = self.check_points(dice, cat)
#        assert isinstance(diceInt,int)
        #convert dice to np.ndarray of len 5
#        dice=np.empty(shape=5,dtype=int)
#        for ii in range(0,5):
#            dice[-ii]=diceInt%10
#            diceInt=diceInt//10
        
            
#        if np.ma.getmask(self.scores)[cat]:
#            self.scores[cat]=0
            
    def open_cats_mask(self):
        return np.ma.getmask(self.scores).astype(int)
    
    def open_cats(self):
        inds = np.array(list(range(len(self.cats))))
#        lp(inds)
#        lp(self.scores.mask)
        return inds[self.scores.mask]
#        return np.maA.getmask(self.scores).astype(int)

    def getUpperSum(self):
        uSum=np.sum(self.scores[:6])
        if np.ma.is_masked(uSum):
            uSum=0
        if uSum>=63:
            uSum+=35
        return uSum
    def getLowerSum(self):
        lSum=np.sum(self.scores[6:])
        if np.ma.is_masked(lSum):
            lSum=0
        return lSum
    def getSum(self):
        return self.getUpperSum()+self.getLowerSum()
    
    def print(self):
        print('='*10 + ' Score Board: ' + '='*10)
#        print()
        for ii in range(13):
            if ii == 6:
                print('-'*34)
            score = '--' if self.scores.mask[ii] else str(self.scores[ii])
            print('{:16}: {:2}'.format(ScoreBoard.cats[ii], score))
        print('='*34)
        print(' '*10 + 'Score: ' + str(self.getSum()) + ' '*10)
        print('='*34)


class Game:
    """Stores a complete game"""
    
    def __init__(self, players):
        """Plays and stores a Yahtzee game.
        
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
        players : Player or list of players with len 13*3
            Artificial player, subclass of AbstractPlayer
        Returns
        -------
        Game
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
#        if isinstance(player, AbstractPlayer):
#            playerGen = lambda: player
#        else:
#            assert isinstance(player, PlayerEnsemble)
#            playerGen = lambda: player.rand()
        if not isinstance(players, list):
            players = [players] * 13*3
        assert len(players) == 3*13, str(players)
        player = iter(players)
        
        self.log = []
        sb = ScoreBoard()
        for cc in range(0,13):
            roundLog = [sb]
            
            dice = Dice()
            deci = next(player).choose_roll(sb, dice, 0)
            roundLog += [dice.copy(), deci]
            
            dice.roll(deci)
            deci = next(player).choose_roll(sb, dice, 1)
            roundLog += [dice.copy(), deci]
            
            dice.roll(deci)
            deci = next(player).choose_cat(sb, dice)
            sb.add(dice, deci)
            roundLog += [dice.copy(), deci]
            
            self.log += [roundLog]
        self.sb = sb
    
#    # trivial iteration for lst = list(game) functionality
#    # https://www.programiz.com/python-programming/iterator
#    def __iter__(self):
#        self.isNext = True
#        return self
#    def __next__(self):
#        if self.isNext:
#            self.isNext = False
#            return self
#        else:
#            raise StopIteration
        
    def print(self):
        
        # sort log by categories
        dfLog = pd.DataFrame(
                self.log, columns=['scores',
                                   'dice0', 'deci0',
                                   'dice1', 'deci1',
                                   'dice2', 'deci2'])
        dfLog.index.name ='round'
        dfLog = dfLog.reset_index()

        dfLog =dfLog.set_index('deci2')

        dfLog = dfLog.sort_index()

        n = 36
        print('='*n + ' Score Board ' + '='*n)
        print('{:16}: {:5} | round - dice (r = reroll)'
              .format('Category', 'Score'))
        print('-'*(2*n + 13))
        
        for ii in range(13):
            if ii == 6:
                print('-'*(2*n+13))
            score = str(self.sb.scores[ii])
            line = '{:16}: {:5} | {:>5} - '.format(ScoreBoard.cats[ii], score,
                    str(dfLog.loc[ii, 'round']))
            for rr in [0,1]:
                dice = dfLog.loc[ii, 'dice' + str(rr)]
                deci = dfLog.loc[ii, 'deci' + str(rr)]
                line += '{:} -> '.format(dice.to_str(deci))
            line += dfLog.loc[ii, 'dice2'].to_str()
            print(line)
        
        print('='*(2*n+13))
        print(' '*(n+0) + ' Score: {:5d}'.format(self.sb.getSum()))
        print('='*(2*n+13))
    
    @property
    def score(self):
        return self.sb.getSum()