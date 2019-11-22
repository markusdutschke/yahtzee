#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:50:06 2019

@author: user
"""
import sys; sys.path.append('./lib/')
from comfct.debug import lp
from yahtzee import Dice, ScoreBoard



def diceToBinary(dice):
    #diceInt integer with 5 digits, each between 1 and 6
    #return 13 digit integer np.ndarray with 0s and 1s
    #dice np.ndarray of len 5
    #strBase6 = ''.join([str(dig-1) for dig in dice])
    #diceInt = int(str(strBase6),6) #values from 0 to 7775
    
    assert dice>=11111 and dice<=66666
    diceInt=int(str(dice-11111),6) #integer of base 6 representation of dice
    #print('diceInt:',diceInt) 
    diceBin=np.zeros(shape=13,dtype=int) #13 digits cover up to 8192
    diceBinStr=bin(diceInt)[2:]
    #print('diceBinStr:',bin(diceInt),diceBinStr)
    for ii in range(0,len(diceBinStr)):
        diceBin[-ii]=diceBinStr[-ii]
    
    return diceBin

def getCatSelBinInfo( scoreBoard, dice):
    #returns np.ndarray of floats between 0 and 1 with len 28
    # encoding the info for Categrorie Selection NN
    assert isinstance(scoreBoard,ScoreBoard)
    binInfo= np.empty(shape=28)
    binInfo[0:13]=scoreBoard.getOpenCategories()
    binInfo[13]=float(scoreBoard.getUpperSum())/140
    binInfo[14]=float(scoreBoard.getLowerSum())/235
    binInfo[15:]=diceToBinary(dice)
    return binInfo




def main1_playARandomGame():
    d = Dice()
    lp(d)
    d.roll( [True, True, False, False, True])
    lp(d)

    sb = ScoreBoard()
    for cc in range(0,13):
        print('STARTING ROUND NUMBER', cc)
        dice = Dice()
        print('1st dice:', dice)
        dice.roll( [True, True, False, False, True])
        print('2nd dice:', dice)
        dice.roll( [True, False, False, False, False])
        print('3rd dice:', dice)
        sb.add(dice,cc)
        sb.print()
        print()




if __name__== "__main__":
    main1_playARandomGame()