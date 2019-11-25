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
import bot
#(PlayerEnsemble,
#        PlayerRandomCrap, PlayerOneShotHero,
#        PlayerOneShotAI, PlayerOneShortAISmartEnc
#        )



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
    game = Game(PlayerRandomCrap())
    game.print()

def main2_simpleBenchmark():
    print('Benchmark Classic Players:')
    for player in [bot.PlayerRandomCrap(), bot.PlayerOneShotHero()]:
#        m, s = benchmark(player, nGames=100)
        m, s = player.benchmark()
        print('\t{:35} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))
        

def main3_initLearningPlayer():
    print('Benchmark Intelligent Players:')
#    players = [PlayerOneShotAI(), PlayerOneShortAISmartEnc()]
    players = [bot.PlayerAI_1SEnc_1(), bot.PlayerAI_1SEnc_2()]
    
    nGames = [1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3]#, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5]
    for nT in nGames:
        nT = int(nT)

        for player in players:
            trainerEnsemble = bot.PlayerEnsemble([
                    (2, player),
                    (1, bot.PlayerRandomCrap()),
                    (1, bot.PlayerOneShotHero())
                    ])
            player.train(nGames=nT-player.nGames, trainerEnsemble=trainerEnsemble)
            m, s = player.benchmark()
            name = player.name + ' ('+str(player.nGames) + ' games)'
            print('\t{:35} {:.1f} +/- {:.1f}'.format(name+':', m, s))



if __name__== "__main__":
#    main1_playARandomGame()
    main2_simpleBenchmark()
    main3_initLearningPlayer()