#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:53:36 2019

@author: user
"""
import numpy as np

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
            self.vals = np.random.randint(1,6,5)
        else:
            assert len(arr) == 5
            self.vals = np.array(arr)
    
    def roll(self, arr):
        """arr is a boolen array"""
        assert len(arr) == 5
        newVals = np.random.randint(1, 6, np.sum(arr))
        self.vals[arr] = newVals
    
    def __str__(self):
        return ', '.join(['{:.0f}'.format(val) for val in self.vals])
    
    def compress(self):
        """Result of the 5 dice is encoded as a 5 digit integer"""
        pass

class ScoreBoard:
    categories=[
            'Aces','Twos','Threes','Fours','Fives','Sixes',
            'Three Of A Kind','Four Of A Kind','Full House',
            'Small Straight','Large Straight','Yahtzee','Chance']
    def __init__(self):
        self.scores=np.ma.masked_array(np.empty(shape=13),mask=True,dtype=int)
        
    def add(self, dice, posCat):
        """dice: Dice instance; posCat: int index for the category of the sb"""
        #dice: np.ndarray, posCat int
        assert isinstance(posCat,int)
        
        assert np.ma.getmask(self.scores)[posCat]==1 \
        , str(self.scores)+ '\n'+str(self.getOpenCategories()) \
        +'\nposCat='+str(posCat)+'\ntry to add to used category!'
        dice = dice.vals
#        assert isinstance(diceInt,int)
        #convert dice to np.ndarray of len 5
#        dice=np.empty(shape=5,dtype=int)
#        for ii in range(0,5):
#            dice[-ii]=diceInt%10
#            diceInt=diceInt//10
        if posCat>=0 and posCat<=5:
            self.scores[posCat]=np.sum(dice[dice==(posCat+1)])
        elif posCat==6:
            if np.amax(np.bincount(dice))>=3:
                self.scores[posCat]=np.sum(dice)
        elif posCat==7:
            if np.amax(np.bincount(dice))>=4:
                self.scores[posCat]=np.sum(dice)
        elif posCat==8:
            sd=np.sort(np.bincount(dice))
            if (sd[-1]==3 and sd[-2]==2) or sd[-1]==5:
                self.scores[posCat]=25
        elif posCat==9:
            sd=np.bincount(dice)
            lenStraight=0
            for ii in range(0,len(sd)):
                if sd[ii]>0:
                    lenStraight+=1
                else:
                    lenStraight=0
                if lenStraight>=4:
                    self.scores[posCat]=30
                    break
        elif posCat==10:
            sd=np.where(np.bincount(dice)>0,1,0)
            if np.sum(sd)==5 and (not (dice==1).any() or not (dice==6).any()):
                self.scores[posCat]=40
        elif posCat==11:
            if np.amax(np.bincount(dice))==5:
                self.scores[posCat]=50
        elif posCat==12:
            self.scores[posCat]=np.sum(dice)  
        else:
            assert False, 'invalid category position, posCat='+str(posCat)
            
        if np.ma.getmask(self.scores)[posCat]:
            self.scores[posCat]=0
            
    def getOpenCategories(self):
        return np.ma.getmask(self.scores).astype(int)

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
            print('{:16}: {:2}'.format(ScoreBoard.categories[ii], score))
        print('='*34)
        print(' '*10 + 'Score: ' + str(self.getSum()) + ' '*10)
        print('='*34)


class Game:
    """Stores a complete game"""
    
    def __init__(self, fctRoll, fctCat):
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
        log = []
        sb = ScoreBoard()
        for cc in range(0,13):
            dice0 = Dice()
            deci0 = fctRoll(sb, dice, 0)
            dice1 = dice.roll(deci0)
            deci1 = fctRoll(sb, dice, 1)
            dice2 = dice.roll(fctRoll(sb, dice, 1))
            deci2 = fctCat(sb, dice)
            sb.add(dice2, deci2)
            
            log += [(dice0, deci0, dice1, deci1, dice2, deci2)]