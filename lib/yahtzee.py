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
from itertools import product


def roll_dice(nDice=5):
    #dice = np.empty(shape=nDice,dtype=int)
    dice=0
    for ii in range(0,nDice):
        #dice[ii]=random.randint(1,6)
        dice*=10
        dice+=np.random.randint(1,6)
    return dice

class Dice:
    """Stores the result of 0 to 5 dice"""
    def __init__(self, vals=None):
        if vals is None:
            self.vals = np.sort(np.random.randint(1,7,5))
        else:
            assert len(vals) <= 5
            self.vals = np.sort(vals)
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
    
    def roll(self, mask):
        """arr is a boolen array: True->reroll this dice"""
        assert len(mask) == 5, str(mask)
#        lp(self.vals)
#        lp(arr)
        randVals = np.random.randint(1, 7, 5)

        newVals = np.where(mask, randVals, self.vals)

        newVals = np.sort(newVals)
        return Dice(newVals)
#        lp(self.vals)
    
    def keep(self, reroll):
        """reroll: [bool]*5, True means reroll
        returns 0 to 5 dice as Dice object"""
        keepDice = self.vals[np.logical_not(reroll)]
        return Dice(keepDice)
    
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
    def copy(self):
        return deepcopy(self)
    
    @property
    def data(self):
        return self.scores.data
    @property
    def mask(self):
        return self.scores.mask
    
    @property
    def score(self):
        return self.getSum()
    
    
    @classmethod
    def get_cat_points(cls, dice, cat):
        if isinstance(dice, Dice):
            dice = dice.vals
        if len(dice)==0:
            return 0
        assert len(dice) <= 5
        assert isinstance(dice, np.ndarray), str(dice) + '; ' + str(type(dice))

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
        
    
    @classmethod
    def stat_cat_score(cls, dice):
        """Statistical category score
        
        forcasts the statistically EXACT score in each category.
        
        dice are the fixed dice only
        if len(dice.vals) == 5,
        this is equivalent to check_points for all cats simulatenously
        
        returns
        meanComb : np.ndarray shape=(13,)
            mean score in each cat
        semComb : np.ndarray shape=(13,)
            standard error of the mean
        """
        assert 0 <= len(dice.vals) <= 5, str(dice)
#        self.scores=np.ma.masked_array(np.empty(shape=13),mask=True,dtype=int)
        
        nRr = 5-len(dice.vals)
        nCombs = 6**nRr

        yCombs = np.empty(shape=(nCombs, 13))
        for mm, comb in enumerate(product([1, 2, 3, 4, 5, 6], repeat=nRr)):
#            diceNew = np.copy(dice.vals)
            diceNew = Dice(list(comb) + list(dice.vals))
#            diceNew = Dice(diceNew)
#                lp(diceOld, deciRr, comb)
            for cc in range(13):
#                lp(diceNew, cc)
                yCombs[mm, cc] = cls.get_cat_points(diceNew, cc)
        meanComb = np.mean(yCombs, axis=0)
        semComb = np.std(yCombs, axis=0) / nCombs**.5
        
        return meanComb, semComb
    
    
    def check_points(self, dice, cat):
        """Just check how many points one would get by assigning dice
        to category number cat
        dice:Dice
        cat: int.
        """
#        lp(dice, type(dice))
        score = ScoreBoard.get_cat_points(dice, cat)
        
        bonus = 0
        us = self.getUpperSum()
        if us < 63 and cat < 6 and us + score >= 63:
            bonus = 35
        return score, bonus
    
    @classmethod
    def check_points_max(self, cat):
        maxPnts = {0: 5,
                   1: 10,
                   2: 15,
                   3: 20,
                   4: 25,
                   5: 30,
                   6: 30,
                   7: 30,
                   8: 25,
                   9: 30,
                   10: 40,
                   11: 50,
                   12: 30}
        return maxPnts[cat]
    
    
    def add(self, dice, cat):
        """dice: Dice instance; posCat: int index for the category of the sb"""
        #dice: np.ndarray, posCat int
#        assert isinstance(posCat, int)
        assert np.issubdtype(type(cat), np.signedinteger)
        
        assert np.ma.getmask(self.scores)[cat]==1 \
        , str(self.scores)+ '\n'+str(self.open_cats_mask()) \
        +'\nposCat='+str(cat)+'\ntry to add to used category!'
#        dice = dice.vals
        self.scores[cat], bonus = self.check_points(dice, cat)
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
        if np.ma.is_masked(uSum):  # all entries masked
            uSum=0
        return uSum
        
    def getLowerSum(self):
        lSum=np.sum(self.scores[6:])
        if np.ma.is_masked(lSum):  # all entries masked
            lSum=0
        return lSum
    
    def getSum(self):
        uSum = self.getUpperSum()
        lSum = self.getLowerSum()
        bonus = 0
        if uSum >= 63:
            bonus += 35

        return uSum + bonus + lSum
    
#    def print(self):
#        print('='*10 + ' Score Board: ' + '='*10)
##        print()
#        for ii in range(13):
#            if ii == 6:
#                print('-'*34)
#            score = '--' if self.scores.mask[ii] else str(self.scores[ii])
#            print('{:16}: {:2}'.format(ScoreBoard.cats[ii], score))
#        print('='*34)
#        print(' '*10 + 'Score: ' + str(self.getSum()) + ' '*10)
#        print('='*34)
    def print(self):
        print(self)
        
    def __str__(self):
        _str = ''
        _str += ('='*10 + ' Score Board: ' + '='*10 + '\n')
#        print()
        for ii in range(13):
            if ii == 6:
                _str += ('-'*34 + '\n')
            score = '--' if self.scores.mask[ii] else str(self.scores[ii])
            _str += ('{:16}: {:2}'.format(ScoreBoard.cats[ii], score) + '\n')
        _str += ('='*34 + '\n')
        _str += (' '*10 + 'Score: ' + str(self.getSum()) + ' '*10 + '\n')
        _str += ('='*34 + '\n')
        return _str


class Game:
    """Stores a complete game"""
    
    def __init__(self, player):
        """player: AbstracPlayer"""
        self.log = pd.DataFrame(
                columns=['scoreBoard',
                         'dice0', 'deci0', 'info0',
                         'dice1', 'deci1', 'info1',
                         'dice2', 'deci2', 'info2'],
                index=list(range(14)))
        self.autoplay(player)

    
    def autoplay(self, player):
        """Plays and stores a Yahtzee game.
        
        Returns
        -------
        None
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
        sb = ScoreBoard()
        for cc in range(0,13):
            prevSb = sb.copy()
            
            dice0 = Dice()
#            lp(player.name)
            deci0, info0 = player.choose_reroll(sb, dice0, 0)
#            lp(deci0)
#            lp(info0)
#            lp(player.choose_reroll(sb, dice0, 0))
#            lp(type(player.choose_reroll(sb, dice0, 0)))
            dice1 = dice0.roll(deci0)
            deci1, info1 = player.choose_reroll(sb, dice1, 1)
            # choose cat
            dice2 = dice1.roll(deci1)
            deci2, info2 = player.choose_cat(sb, dice2)
            sb.add(dice2, deci2)
#            lp(self.log)
            self.log.loc[cc, :] = [
                    prevSb,
                    dice0, deci0, info0, dice1, deci1, info1,
                    dice2, deci2, info2]
        self.log.loc[13, 'scoreBoard'] = sb

#        self.finalScoreBoard = sb
    

        
    def __str__(self, debugLevel=0):
        _str = ''
        # sort log by categories
#        lp(self.log)
#        dfLog = pd.DataFrame(
#                self.log, columns=['scores',
#                                   'dice0', 'deci0',
#                                   'dice1', 'deci1',
#                                   'dice2', 'deci2', 'info2'])
        dfLog = self.log
#        lp(dfLog)
        dfLog.index.name ='round'
        sb = self.log.loc[13, 'scoreBoard']
        
        if debugLevel >= 1:
            for ii in range(13):
                _str += 'ROUND ' + str(ii) +'\n'
                _str += 'Dice0: ' + str(dfLog.loc[ii,'dice0']) +'\n'
                _str += 'Deci0: ' + str(dfLog.loc[ii,'deci0']) +'\n'
                _str += 'Info0: ' + str(dfLog.loc[ii,'info0']) +'\n'
                _str += '--\n'
                _str += 'Dice1: ' + str(dfLog.loc[ii,'dice1']) +'\n'
                _str += 'Deci1: ' + str(dfLog.loc[ii,'deci1']) +'\n'
                _str += 'Info1: ' + str(dfLog.loc[ii,'info1']) +'\n'
                _str += '--\n'
                _str += 'Dice2: ' + str(dfLog.loc[ii,'dice2']) +'\n'
                _str += 'Deci2: ' + str(ScoreBoard.cats[dfLog.loc[ii,'deci2']]) +'\n'
                _str += 'Info2: ' + str(dfLog.loc[ii,'info2']) +'\n'
                _str += '===============\n'
#                for jj in range(3):
#                _str += (
#                        '--\n'
#                        + 'DICE: ' + str(dfLog.loc[ii,'dice2'])
#                        + ';\nEVAL: ' + str(dfLog.loc[ii,'info2'])
#                        + ';\nDECISION: ' + ScoreBoard.cats[dfLog.loc[ii,'deci2']]
#                        )
#                _str += '\n'
        
        dfLog = dfLog.reset_index()
        dfLog =dfLog.set_index('deci2')
        dfLog = dfLog.sort_index()

        n = 36
        _str += ('='*n + ' Score Board ' + '='*n +'\n')
        _str += ('{:16}: {:5} | round - dice (r = reroll)\n'
              .format('Category', 'Score'))
        _str += ('-'*(2*n + 13) +'\n')
        
        for ii in range(13):
            
            if ii == 6:
                _str += ('-'*(2*n+13) +'\n')
                upperSum = sb.getUpperSum()
                _str += '{:16}: {:5} |\n'.format(
                        'Upper Sum', str(upperSum))
                _str += '{:16}: {:5} |\n'.format(
                        'Bonus', '35' if upperSum >= 63 else '--')
                _str += ('-'*(2*n+13) +'\n')
            score = str(sb.scores[ii])
            line = '{:16}: {:5} | {:>5} - '.format(ScoreBoard.cats[ii], score,
                    str(dfLog.loc[ii, 'round']))
            for rr in [0,1]:
                dice = dfLog.loc[ii, 'dice' + str(rr)]
                deci = dfLog.loc[ii, 'deci' + str(rr)]
                line += '{:} -> '.format(dice.to_str(deci))
            line += dfLog.loc[ii, 'dice2'].to_str()
            _str += (line +'\n')
        
        _str += ('='*(2*n+13) +'\n')
        _str += (' '*(n+0) + ' Score: {:5d}\n'.format(sb.getSum()))
        _str += ('='*(2*n+13) +'\n')
        return _str
    
    @property
    def score(self):
        return self.log.loc[13, 'scoreBoard'].getSum()



class Game_depricated:
    """Stores a complete game"""
    
    def __init__(self, players=None):
        
        self.catLog = []
        if not players is None:
            self.log = []
            self.autoplay(players)
        else:
            self.sb = ScoreBoard()
            self.attempt = 0
            self.dice = Dice()
            self.waitForAction = False
        
        
            self.log = [[self.sb.copy(), self.dice.copy()]]
            
            
    def ask_action(self):
        assert self.waitForAction == False
        
        if len(self.sb.open_cats()) == 0:
            return self.sb.getSum()
        
        if self.attempt < 2:
            act = 'choose_dice'
            paras = (self.sb, self.dice, self.attempt)
        else:
            assert self.attempt == 2
            act = 'choose_cat'
            paras = (self.sb, self.dice)
        self.waitForAction = True
        return act, paras
    
    def perf_action(self, act, para):
        assert self.waitForAction == True
        
        if self.attempt < 2:
            assert act == 'choose_dice'
            self.dice.roll(para)
            self.log[-1] += [para, self.dice.copy()]
        else:
            assert self.attempt == 2
            assert act == 'choose_cat'
            self.catLog += [(self.sb.copy(), self.dice, para)]
            self.sb.add(self.dice, para)
            self.dice = Dice()
            self.log[-1] += [para, 'no Info']
            if len(self.sb.open_cats()) > 0:
                self.log += [[self.sb.copy(), self.dice.copy()]]
            
            
        self.attempt = (self.attempt+1)%3
        self.waitForAction = False
        

    
    def autoplay(self, players):
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
        
#        self.log = []
#        self.catLog = []  # log for cat decision learning
        sb = ScoreBoard()
        for cc in range(0,13):
            roundLog = [sb.copy()]
            
            dice = Dice()
            deci = next(player).choose_reroll(sb, dice, 0)
            roundLog += [dice.copy(), deci]
            
            dice.roll(deci)
            deci = next(player).choose_reroll(sb, dice, 1)
            roundLog += [dice.copy(), deci]
            
            # choose cat
            dice.roll(deci)
#            deci, info = next(player).choose_cat(sb, dice, debugLevel=1)
            plyr = next(player)
            deci = plyr.choose_cat(sb, dice)
            try:
                opts = plyr.eval_options_cat(sb, dice)
#                opts = plyr.eval_cats(sb, dice)
                info = '\n\t' + '\n\t'.join([
                        '{:}: {:.2f} + {:.2f} = {:.2f}'.format(
                                ScoreBoard.cats[cat], dirRew, futRew, rew)
                        for cat, rew, dirRew, futRew in opts])
            except:
                info = 'No info for ' + plyr.name
            self.catLog += [(sb.copy(), dice, deci)]
            sb.add(dice, deci)
            roundLog += [dice.copy(), deci, info]
            
            
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
        
    def print_depricated(self):
        
        # sort log by categories
        dfLog = pd.DataFrame(
                self.log, columns=['scores',
                                   'dice0', 'deci0',
                                   'dice1', 'deci1',
                                   'dice2', 'deci2', 'info2'])
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
        
    def __str__(self, debugLevel=0):
        _str = ''
        # sort log by categories
#        lp(self.log)
        dfLog = pd.DataFrame(
                self.log, columns=['scores',
                                   'dice0', 'deci0',
                                   'dice1', 'deci1',
                                   'dice2', 'deci2', 'info2'])
#        lp(dfLog)
        dfLog.index.name ='round'
        
        if debugLevel >= 1:
            for ii in range(13):
                _str += (
                        '--\n'
                        + 'DICE: ' + str(dfLog.loc[ii,'dice2'])
                        + ';\nEVAL: ' + str(dfLog.loc[ii,'info2'])
                        + ';\nDECISION: ' + ScoreBoard.cats[dfLog.loc[ii,'deci2']]
                        )
                _str += '\n'
        
        dfLog = dfLog.reset_index()
        dfLog =dfLog.set_index('deci2')
        dfLog = dfLog.sort_index()
        
#        lp(dfLog['info2'])
        
        
                

        n = 36
        _str += ('='*n + ' Score Board ' + '='*n +'\n')
        _str += ('{:16}: {:5} | round - dice (r = reroll)\n'
              .format('Category', 'Score'))
        _str += ('-'*(2*n + 13) +'\n')
        
        for ii in range(13):
            if ii == 6:
                _str += ('-'*(2*n+13) +'\n')
                upperSum = self.sb.getUpperSum()
                _str += '{:16}: {:5} |\n'.format(
                        'Upper Sum', str(upperSum))
                _str += '{:16}: {:5} |\n'.format(
                        'Bonus', '35' if upperSum >= 63 else '--')
                _str += ('-'*(2*n+13) +'\n')
            score = str(self.sb.scores[ii])
            line = '{:16}: {:5} | {:>5} - '.format(ScoreBoard.cats[ii], score,
                    str(dfLog.loc[ii, 'round']))
            for rr in [0,1]:
                dice = dfLog.loc[ii, 'dice' + str(rr)]
                deci = dfLog.loc[ii, 'deci' + str(rr)]
                line += '{:} -> '.format(dice.to_str(deci))
            line += dfLog.loc[ii, 'dice2'].to_str()
            _str += (line +'\n')
        
        _str += ('='*(2*n+13) +'\n')
        _str += (' '*(n+0) + ' Score: {:5d}\n'.format(self.sb.getSum()))
        _str += ('='*(2*n+13) +'\n')
        return _str
    
    @property
    def score(self):
        return self.sb.getSum()