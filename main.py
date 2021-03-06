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
import datetime as dt
import pickle

# Benchmark games should not overlap with training games
BENCHMARK_SEED = 618225912

#(PlayerEnsemble,
#        PlayerRandomCrap, PlayerOneShotHero,
#        PlayerOneShotAI, PlayerOneShortAISmartEnc
#        )



#def diceToBinary(dice):
#    #diceInt integer with 5 digits, each between 1 and 6
#    #return 13 digit integer np.ndarray with 0s and 1s
#    #dice np.ndarray of len 5
#    #strBase6 = ''.join([str(dig-1) for dig in dice])
#    #diceInt = int(str(strBase6),6) #values from 0 to 7775
#    
#    assert dice>=11111 and dice<=66666
#    diceInt=int(str(dice-11111),6) #integer of base 6 representation of dice
#    #print('diceInt:',diceInt) 
#    diceBin=np.zeros(shape=13,dtype=int) #13 digits cover up to 8192
#    diceBinStr=bin(diceInt)[2:]
#    #print('diceBinStr:',bin(diceInt),diceBinStr)
#    for ii in range(0,len(diceBinStr)):
#        diceBin[-ii]=diceBinStr[-ii]
#    
#    return diceBin
#
#def getCatSelBinInfo( scoreBoard, dice):
#    #returns np.ndarray of floats between 0 and 1 with len 28
#    # encoding the info for Categrorie Selection NN
#    assert isinstance(scoreBoard,ScoreBoard)
#    binInfo= np.empty(shape=28)
#    binInfo[0:13]=scoreBoard.getOpenCategories()
#    binInfo[13]=float(scoreBoard.getUpperSum())/140
#    binInfo[14]=float(scoreBoard.getLowerSum())/235
#    binInfo[15:]=diceToBinary(dice)
#    return binInfo





#def main1_playARandomGame():
#    game = Game(bot.PlayerRandomCrap())
#    game.print()
#
#def main2_simpleBenchmark():
#    print('Benchmark Classic Players:')
#    for player in [bot.PlayerRandomCrap(),
#                   bot.PlayerOneShotHero(),
#                   bot.Player1ShotMarkus(),
##                   bot.Player1ShotMonteCarlo(),
#                   ]:
##        m, s = benchmark(player, nGames=100)
#        m, s = player.benchmark()
#        print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))
#        
#
#def main3_initLearningPlayer():
#    print('Benchmark Intelligent Players:')
##    players = [PlayerOneShotAI(), PlayerOneShortAISmartEnc()]
#    players = [
##            bot.PlayerOneShotAI_v2(),
##            bot.PlayerOneShotAI_new(),
##            bot.PlayerAI_1SEnc_1(),
##            bot.PlayerAI_1SEnc_2(),
##            bot.PlayerAI_1SEnc_3(),
##            bot.PlayerAI_1SEnc_4(),
##            bot.PlayerAI_1SEnc_5(),
##            bot.PlayerAI_1SEnc_6(),
##            bot.Player1ShotMarkus(),
#            bot.PlayerAI_full_v0(),
#               ]
#    
#    nGames = [1, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5]
##    nGames = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
##    nGames = np.arange(1,200,5)
##    nGames = np.arange(1,200,1)
#    nGames = [1, 5, 10, 15, 20]
#    for nT in nGames:
#        nT = int(nT)
#
#        for player in players:
##            trainerEnsemble = bot.PlayerEnsemble([
##                    (10, player),
##                    (1, bot.PlayerRandomCrap()),
###                    (1, bot.PlayerOneShotHero())
##                    ])
##            player.train(nGames=nT-player.nGames, trainerEnsemble=trainerEnsemble)
#            if hasattr(player, 'train'):
#                player.train(nGames=nT-player.nGames)
#            else:
#                player.nGames = 0
##            assert False
#            m, s = player.benchmark(seed=None)
#            name = player.name + ' ('+str(player.nGames) + ' games)'
#            lp('\t{:50} {:.1f} +/- {:.1f}'.format(name+':', m, s))
#            
#            
##            if m > 115:
###            if player.nGames >=6:
##                np.random.seed(0)
##                print(Game(player).__str__(debugLevel=1))
##                print(Game(player).__str__(debugLevel=1))
##                assert False
#
#        print('\t-')
    

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
    lp('Training Intelligent Player:')
    player = bot.PlayerAI_full_v2(fn='./tmp/PlayerAI_full_v2-nGame0.pick')
    
    playerFn = (
            lambda it: './tmp/{:}-nGame{:d}.pick'
            .format(player.name, it))
    mmax = 0

    
    
    nGames = list(range(0,100,10)) + list(range(100,25000,100))
    for nT in nGames:
        nT = int(nT)
        if nT<=player.nGames:
            continue
        player.train(nGames=nT-player.nGames, benchmarkSeed=BENCHMARK_SEED)
        m, s = player.benchmark(seed=BENCHMARK_SEED)
        mmax = max(m, mmax)
        name = player.name + ' ('+str(player.nGames) + ' games)'
        strTime = dt.datetime.now().strftime('%H:%M')
        print('\t{:5}   {:32} {:.1f} +/- {:.1f}\tmax: {:.1f}'
              .format(strTime, name+':', m, s, mmax))
        
        player.save(playerFn(player.nGames))
        


def main6_playAGame():
    print('Check out some games:')
    player = bot.PlayerAI_full_v2(fn='./tmp/PlayerAI_full_v2-nGame8000.pick')
    
#    lp(player.predict_Ex(Dice([5,5,5,5,5]),[True, True, False, False, False]))
#    lp(player.predict_Ex(Dice([1,2,3,4,5]),[True, True, False, True, False]))
    
    if False:  # check rgrEx
        testSets = [[1,2,3], [1,2,3,4], [5,5], [2,2,3,3], [2,2,2,3], [2,2,2,2],
                        [3,3,3]]
        for keepDice in testSets:
            nMiss = 5 - len(keepDice)
            pred = player.predict_Ex(Dice(keepDice+[5]*nMiss),
                                     [False]*len(keepDice)+[True]*nMiss)
            ex, std = ScoreBoard.stat_cat_score(Dice(keepDice))
            lp('predictions for situation:', str(keepDice))
            print('\t{:20} {:} {:}'.format('Combination', 'Pred', 'Exact'))
            for ii in range(13):
                print('\t{:20} {:.2f} {:.2f}'.format(ScoreBoard.cats[ii], pred[ii], ex[ii]))
    
    np.random.seed(6)
    for ii in range(1):
        game = Game(player)
        print(game.__str__(debugLevel=1))


def dev7_benchmark_v1Ex():
    lp('main7_benchmark_v1Ex()')
    
    
    nTrainings = [1e3, 1e4, 5e4, 1e5, 5e5]
#    nTrainings = [1e1, 1e2, 5e2, 1e3]
#    nTrainings = [1e5]
    for nt in nTrainings:
        nt = int(nt)
        player = bot.PlayerAI_full_v1()
        player.aux_Ex_train(n=nt, optRgrParas=False)
        m, s = player.aux_Ex_benchmark(n=1000)
#        lp(nt, ':', m, s)
        lp('benchmark for', nt, 'games:')
        for ii in range(13):
            print('\t{:20} {:.2f} {:.2f}'.format(ScoreBoard.cats[ii], m[ii], s[ii]))
        
#        lp(player.predict_Ex(Dice([1,2,3]+[5,5]), [False]*3+[True]*2))
#        lp(player.predict_Ex(Dice([1,2,3,5]+[5]), [False]*4+[True]*1))
#        lp(player.predict_Ex(Dice([2,2,3,3]+[5]), [False]*4+[True]*1))
#        lp(player.predict_Ex(Dice([5,5,5,5]+[5]), [False]*4+[True]*1))
#        lp(player.predict_Ex(Dice([1]+[2,2,4,5]), [False]*1+[True]*4))
        
        testSets = [[1,2,3], [1,2,3,4], [5,5], [2,2,3,3], [2,2,2,3], [2,2,2,2],
                    [3,3,3]]
        for keepDice in testSets:
            nMiss = 5 - len(keepDice)
            pred = player.predict_Ex(Dice(keepDice+[5]*nMiss),
                                     [False]*len(keepDice)+[True]*nMiss)
            ex, std = ScoreBoard.stat_cat_score(Dice(keepDice))
            lp('predictions for situation:', str(keepDice))
            for ii in range(13):
                print('\t{:20} {:.2f} {:.2f}'.format(ScoreBoard.cats[ii], pred[ii], ex[ii]))
        
        print()
        print()

def debug8_findMemLeak():
    
    def sizeKb(obj):
        return len(pickle.dumps(obj))/1024
    
    def sizeKbAttrs(obj):
        dct = {}
        for att in obj.__dict__.keys():
            dct[att] = sizeKb(obj.__dict__[att])
        return dct
    
    def inspect(obj):
        dct = sizeKbAttrs(obj)
        _str = ''
        for att in dct.keys():
            _str += '{:}: {:.2f}\n'.format(att, dct[att])
        return _str


    lp('debug8_findMemLeak')
    player = bot.PlayerAI_full_v1()
    nGames = [2, 3, 30, 40, 50, 100, 200, 300, 400]
    for nT in nGames:
        nT = int(nT)
        if nT<=player.nGames:
            continue
        player.train(nGames=nT-player.nGames)
        
        player.save('./tmp/test'+str(nT)+'.pick')
        
        a, b = sizeKb(player.rgrSC), sizeKb(player.rgrSC.loss_curve_)
        lp('{:d} games, {:.2f}, {:.2f}, {:.2f}'.format(player.nGames, a, b, a-b))
        lp(player.rgrSC.loss_curve_)
        lp(inspect(player.rgrSC))
        
#def main9_rapidPrototype():
#    lp('Training Intelligent Player:')
#    player = bot.PlayerAI_full_v2(fn='./tmp/PlayerAI_full_v2-nGame0.pick')
#    
#    playerFn = (
#            lambda it: './tmp/{:}-nGame{:d}.pick'
#            .format(player.name, it))
#    mmax = 0
#    
##    loadIter = 0  # 0: OFF
##    try:
##        player.load(playerFn(loadIter))
##    except FileNotFoundError:
##        print('No player model saved. Starting Training from zero ...')
##    else:
##        print('Loaded player from file:', playerFn(loadIter))
##        m, s = player.benchmark(seed=None)
##        name = player.name + ' ('+str(player.nGames) + ' games)'
##        lp('\t{:50} {:.1f} +/- {:.1f}'.format(name+':', m, s))
#    
#    
#    nGames = [1,2,3,4,5]
#    for nT in nGames:
#        nT = int(nT)
#        if nT<=player.nGames:
#            continue
#        player.train(nGames=nT-player.nGames, benchmarkSeed=BENCHMARK_SEED)
##        m, s = player.benchmark(seed=BENCHMARK_SEED, nGames=10, nBins=10)
#        m, s = player.benchmark(seed=BENCHMARK_SEED)
#        mmax = max(m, mmax)
#        name = player.name + ' ('+str(player.nGames) + ' games)'
#        strTime = dt.datetime.now().strftime('%H:%M')
#        print('\t{:5}   {:32} {:.1f} +/- {:.1f}\tmax: {:.1f}'
#              .format(strTime, name+':', m, s, mmax))
#        
##        player.save(playerFn(player.nGames))
#        
##        assert m < 200, 'Found nice result'
        



def demo():
    print('='*80 + '\n' +
          'Lets first have a look at the final performance of the trained AI:'
          + '\n' + '='*80)
    print()
    
    print('AI-Players created by Markus Dutschke:')
    print()
    print('\t{:50} {:}'.format('Description', 'avg. Score'))
    print('\t' + '-'*80)
    lstPlayersAI = []
    lstPlayersAI += [
            bot.PlayerAI_full_v0(
                    fn='./trainedBots/PlayerAI_full_v0-nGame1053.pick'),
            bot.PlayerAI_full_v1(
                    fn='./trainedBots/PlayerAI_full_v1-nGame5700.pick'),
            bot.PlayerAI_full_v2(
                    fn='./trainedBots/PlayerAI_full_v2-nGame8000.pick'),
                    ]
    for player in lstPlayersAI:
        m, s = player.benchmark(seed=BENCHMARK_SEED)
        print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))
    print()
    
    print('A view benchmarks for comparison (check papers in README):')
    print()
    print('\t{:50} {:}'.format('Description', 'avg. Score'))
    print('\t' + '-'*80)
    lstPlayersAlg = []
    lstPlayersAlg += [
            bot.PlayerAlg_oneShot_greedy(),
            bot.PlayerAlg_full_greedy(),
                    ]
    for player in lstPlayersAlg:
        m, s = player.benchmark(seed=BENCHMARK_SEED)
        print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))
    print()
    print('\t{:50} {:.1f}'.format('Random, no Bonus (Verhoff):', 45.95))
    print('\t{:50} {:.1f}'.format('Greedy (Glenn06):', 218.05))
    print('\t{:50} {:.1f}'.format('Greedy (Felldin):', 221.68))
    print('\t{:50} {:.1f}'.format('Optimal Strategy (Holderied):', 245.9))
    print()
    print('\t{:50} {:.1f}'.format('Human trials, group 1, 26 games:', 239.5))
    print('\t{:50} {:.1f}'.format('Human trials, group 2, 8 games:', 202.8))
    
    

    
    print()
    print()
    
    
    print('='*80 + '\n' +
          'Okay, lets check out a few games of the best AI Player!'
          + '\n' + '='*80)
    print()
    player = lstPlayersAI[-1]
    seeds = [3, 6, 9]
    for ii in range(len(seeds)):
        np.random.seed(seeds[ii])
        game = Game(player)
        print(game)
    print()
    print()
    
    
    print('='*80 + '\n' +
          'Now, let\'s see how such a cool AI player is trained ...'
          + '\n' + '='*80)
    print()
    print('Note: training + benchmarks takes a few hours')
    player = bot.PlayerAI_full_v2()
    nGames = (
            list(range(1,10,1))
            + list(range(10,101,10))
            + list(range(100,8001,100))
            )
    print()
    print('\t{:20} {:}'.format('# Trainings', 'Score'))
    for nT in nGames:
        nT = int(nT)
        if nT<=player.nGames:
            continue
        player.train(nGames=nT-player.nGames)
        m, s = player.benchmark(seed=BENCHMARK_SEED)
        print('\t{:20} {:.1f} +/- {:.1f}'.format(str(player.nGames), m, s))
    player.save('./trainedBots/PlayerAI_full_v2-nGame8000-2.pick')
        

# --- ToDo
"""
Lets list some todos here, to improve this code even further 
or just to check out some ideas

- try out some bonus encoding:
    The question is a bit how. If it encodes a clear recommendation to MLPrgr
    it should be exact! Maybe try the absolute upper some and some fancy
    encoded bonus as redundant information.
    Normalize the bonus information to range [0,1] due to mlprgr properties

- check out the average score of training games and 
    compare if this is somehow close to the benchmark scores

- set gamma=0.98 and test.
    No idea, what this is going to help, but it is done in literature,
    so it cant hurt much here. One should evaluate the bene/male-fit carefully.

- implement some exploartion strategy with is increasing the minMaxRatio with a
    higher score.
    
    Just like: 
        minMaxRatio = avgQ /220 * 1000
    or even:
        minMaxRatio = exp(avgQ /220) * 1000
        
    minMaxRatio should also increase with increasing game round.
    It is more important to explore moves in the early stages of the game,
    as forecasts are less exact then.

    Use this as a separate exploration strategy. This is too far from softmax.

- move bonus consideration to future reward (rgrSC) and not the direct reward
    Should  avoid different scaling of scores problem
    Should avoid greedy get bonus behavior for easy reachable bonus scenarios

- increase NN size

- add a Qt GUI for using this programm as a cheat tool in real yahtzee games
    - textfields for current score board, which are automatically updated
    - one textfiled for current (compressed) dice combination.
        Example: 11632
"""


if __name__ == "__main__":
    np.random.seed(0)
    demo()

#    main5_trainFullAIPlayer()
#    main6_playAGame()
