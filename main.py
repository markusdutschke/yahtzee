#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:50:06 2019

@author: user
"""
import numpy as np
import sys; sys.path.append('./lib/')
from comfct.debug import lp
from yahtzee import Dice, ScoreBoard, Game
from artificial_player import benchmark, PlayerRandom, PlayerOneShotHero



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
    game = Game(PlayerRandom())
    game.print()

def main2_simpleBenchmark():
    print('Benchmarking players:')
    for player in [PlayerRandom(), PlayerOneShotHero()]:
        m, s = benchmark(player, nIter=100)
        print('\t{:30} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))





if __name__== "__main__":
#    main1_playARandomGame()
    main2_simpleBenchmark()