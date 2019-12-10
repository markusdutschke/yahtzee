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
from sklearn.neural_network import MLPRegressor
import bot

# Benchmark games should not overlap with training games
BENCHMARK_SEED = 618225912

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
    game = Game(bot.PlayerRandomCrap())
    game.print()

def main2_simpleBenchmark():
    print('Benchmark Classic Players:')
    for player in [bot.PlayerRandomCrap(),
                   bot.PlayerOneShotHero(),
                   bot.Player1ShotMarkus(),
#                   bot.Player1ShotMonteCarlo(),
                   ]:
#        m, s = benchmark(player, nGames=100)
        m, s = player.benchmark()
        print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))
        

def main3_initLearningPlayer():
    print('Benchmark Intelligent Players:')
#    players = [PlayerOneShotAI(), PlayerOneShortAISmartEnc()]
    players = [
#            bot.PlayerOneShotAI_v2(),
#            bot.PlayerOneShotAI_new(),
#            bot.PlayerAI_1SEnc_1(),
#            bot.PlayerAI_1SEnc_2(),
#            bot.PlayerAI_1SEnc_3(),
#            bot.PlayerAI_1SEnc_4(),
#            bot.PlayerAI_1SEnc_5(),
#            bot.PlayerAI_1SEnc_6(),
#            bot.Player1ShotMarkus(),
            bot.PlayerAI_full_v0(),
               ]
    
    nGames = [1, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5]
#    nGames = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#    nGames = np.arange(1,200,5)
#    nGames = np.arange(1,200,1)
    nGames = [1, 5, 10, 15, 20]
    for nT in nGames:
        nT = int(nT)

        for player in players:
#            trainerEnsemble = bot.PlayerEnsemble([
#                    (10, player),
#                    (1, bot.PlayerRandomCrap()),
##                    (1, bot.PlayerOneShotHero())
#                    ])
#            player.train(nGames=nT-player.nGames, trainerEnsemble=trainerEnsemble)
            if hasattr(player, 'train'):
                player.train(nGames=nT-player.nGames)
            else:
                player.nGames = 0
#            assert False
            m, s = player.benchmark(seed=None)
            name = player.name + ' ('+str(player.nGames) + ' games)'
            lp('\t{:50} {:.1f} +/- {:.1f}'.format(name+':', m, s))
            
            
#            if m > 115:
##            if player.nGames >=6:
#                np.random.seed(0)
#                print(Game(player).__str__(debugLevel=1))
#                print(Game(player).__str__(debugLevel=1))
#                assert False

        print('\t-')
    

def main4_evaluateModels():
    for model in [
#            (bot.PlayerOneShotAI_new,
#             dict(mlpRgrArgs={'hidden_layer_sizes':(40, 50, 40, 25, 20, 10)})),
#             (bot.PlayerOneShotAI_new,
#             dict(mlpRgrArgs={'hidden_layer_sizes':(30, 20, 15)})),
#             bot.PlayerOneShotHero,
             bot.Player1ShotMarkus,
#             bot.Player1ShotMonteCarlo,
#             (bot.PlayerOneShotAI_new, {}),
             (bot.PlayerOneShotAI_v2, {}),
#            bot.PlayerAI_1SEnc_1,
#            bot.PlayerAI_1SEnc_2,
#            bot.PlayerAI_1SEnc_3,
#            bot.PlayerAI_1SEnc_4,
#            bot.PlayerAI_1SEnc_5,
#            bot.PlayerAI_1SEnc_6,
            ]:
        kwargs = {}
        if isinstance(model, tuple):
            model, kwargs = model
        print('\n\n'+model.name)
        
        nGames = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        nGames = [1, 5, 10, 15, 20]
        nGames = [1, 2, 3]
        
        df = model.modelBenchmark(nGames=nGames, **kwargs)
        print(df)
#        df = model.modelBenchmark(
#                nGames=[1, 2],
#                **kwargs)
#        print(df)


def main5_trainFullAIPlayer():
    print('Training Intelligent Player:')
    player = bot.PlayerAI_full_v0()
    
    playerFn = (
            lambda it: './trainedBots/{:}-nGame{:d}.pick'
            .format(player.name, it))
    
    loadIter = 0  # 0: OFF
    try:
        player.load(playerFn(loadIter))
    except FileNotFoundError:
        print('No player model saved. Starting Training from zero ...')
    else:
        print('Loaded player from file:', playerFn(loadIter))
        m, s = player.benchmark(seed=None)
        name = player.name + ' ('+str(player.nGames) + ' games)'
        lp('\t{:50} {:.1f} +/- {:.1f}'.format(name+':', m, s))
    
    
    nGames = list(range(0,50000,100))
    nGames = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 200, 500] + list(range(900,1400,1))
    
    for nT in nGames:
        nT = int(nT)
        if nT<=player.nGames:
            continue
        player.train(nGames=nT-player.nGames)
        m, s = player.benchmark(seed=BENCHMARK_SEED)
        name = player.name + ' ('+str(player.nGames) + ' games)'
        lp('\t{:50} {:.1f} +/- {:.1f}'.format(name+':', m, s))
        
        player.save(playerFn(player.nGames))
        
        assert m < 200, 'Found nice result'


def main6_playAGame():
    print('Check out some games:')
    player = bot.PlayerAI_full_v0()
    
    playerFn = (
            lambda it: './trainedBots/{:}-nGame{:d}.pick'
            .format(player.name, it))
    
    loadIter = 2400  # 0: OFF
    try:
        player.load(playerFn(loadIter))
    except FileNotFoundError:
        print('No player model saved. Starting Training from zero ...')
    else:
        print('Loaded player from file:', playerFn(loadIter))
    
    for ii in range(3):
        game = Game(player)
        print(game)

def demo():
#    print('LETS FIRST HAVE A LOOK AT THE FINAL PERFORMANCE OF THIS CODE:')
    print('='*80 + '\n' +
          'Lets first have a look at the final performance of the trained AI:'
          + '\n' + '='*80)
    print()
#    print('Benchmark: Classic Players')
#    print('\t-no use of any machine learning')
#    print('\t-no re-rolling of any dice.'
#    print('\t The result is directly entered in a category on the score board')
#    print('\tOne Shot Human is a collection of human \'strategies\' '
#          'without re-rolling.')
#    for player in [bot.PlayerRandomCrap(),
#                   bot.PlayerOneShotHero(),
#                   bot.Player1ShotHuman(),
#                   ]:
##        m, s = benchmark(player, nGames=100)
#        m, s = player.benchmark(seed=0)
#        print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))
    print('A view benchmarks:')
    print()
    print('\t{:50} {:}'.format('Description', 'avg. Score'))
    print('\t' + '-'*80)
    print('\t{:50} {:.1f}'.format('Random actions:', 43))
    print('\t{:50} {:.1f}'.format('Greedy strategy without re-rolls:', 111))
    print('\t{:50} {:.1f}'.format('Experiences Human Player (TARGET):', 250))
    print()
    
    print('Benchmark: AI-Players')
    lstPlayers = []
    lstPlayers += [
            bot.PlayerAI_full_v0(
                    fn='./trainedBots/PlayerAI_full_v0-nGame1100.pick'),
                    ]
    for player in lstPlayers:
        m, s = player.benchmark(seed=BENCHMARK_SEED)
        print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))
    
    print()
    print()
    print('='*80 + '\n' +
          'Okay, lets check out a few games of this fancy AI!'
          + '\n' + '='*80)
    print()
    
    for ii in range(3):
        game = Game(lstPlayers[0])
        print(game)

if __name__== "__main__":
    np.random.seed(0)
#    demo()
    
#    main1_playARandomGame()
#    main2_simpleBenchmark()
#    main3_initLearningPlayer()
#    main4_evaluateModels()
    
    main5_trainFullAIPlayer()
#    main6_playAGame()
    
