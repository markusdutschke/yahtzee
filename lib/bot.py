#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:02:57 2019

@author: user
"""
from comfct.debug import lp
from abc import ABC, abstractmethod
import numpy as np
import random
import pandas as pd
from yahtzee import Game, ScoreBoard, Dice
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_is_fitted
from comfct.list import list_cast, split_in_consec_ints
from comfct.numpy import weighted_choice, arreq_in_list
#from tqdm import tqdm
from progressbar import progressbar
#from progress.bar import Bar
import warnings
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.validation import check_is_fitted
from itertools import product
import pickle
from copy import deepcopy
#from itertools import product


def explore_boltzmann(actions, qs, temp=None, minMaxRat=None):
    """Performing Bolzmann weighing (also called softmax).
    
    Each possible action gets a probabilistic weight of exp(q / temp).
    An action is chosen by those probabilities.
    This weighting is quite random in the early rounds of a game
    (all qs similar and high) and very deterministic at the final rounds
    (qs are small and quite different).
    
    Parameters
        ----------
    actions : list 
        actions to choose from. Sorted by corresponding q
    qs : list of floats
        same length as actions, cooresponding values of the Q-Function
    temp : float
        Boltzmann temperature
    minMaxRat : float >= 1
        Probability ratio of the most favorable and the most unfavorable action
        
    Returns
        -------
    action : one elemt of actions
    """
    assert temp is None or minMaxRat is None
    qs = np.array(qs)
    if temp is None:
        assert minMaxRat is not None
        probs = qs - np.amin(qs)
        if np.amax(probs) == 0:
            return np.random.choice(actions)
        alpha = np.log(minMaxRat)/np.amax(probs)
        probs = np.exp(alpha*probs)
    else:
        probs = np.exp(qs / temp)
    assert np.isfinite(probs).all(), str(probs) + ';' + str(qs) + '; ' + str(temp)
#    lp(actions)
#    lp(weights)
    return weighted_choice(actions, probs)


def explore_epsgreedy(actions, qs, epsilon):
    """Performing Bolzmann weighing (also called softmax).
    
    Each possible action gets a probabilistic weight of exp(q / temp).
    An action is chosen by those probabilities.
    This weighting is quite random in the early rounds of a game
    (all qs similar and high) and very deterministic at the final rounds
    (qs are small and quite different).
    
    Parameters
        ----------
    actions : list 
        actions to choose from. Sorted by corresponding q
    qs : list of floats
        same length as actions, cooresponding values of the Q-Function
    temp : float
        Boltzmann temperature
        
    Returns
        -------
    action : one elemt of actions
    """
    rndNo = np.random.rand()
    if rndNo <= epsilon:
        action = np.random.choice(actions)
    else:
        return actions[0]


class statCatScorePredictor_exact:
    """Predicts the statistical score in the different categories,
    based on a number of dice, which are kept fixed.
    
    Whenever a new combination of dice is requested, the statistically
    exact probabilities are calculated first and the results are stored in
    a pandas dataframe."""
    lutCols = ['c'+str(ii) for ii in range(13)]  # lookuptable columns
    
    def __init__(self):
        self.lookupTable = pd.DataFrame(
                columns=self.lutCols)
    
    def predict(self, dice):
        """predict the expected score in each cat for given dice"""
        comp = dice.compress()
#        lp(comp)
        if not comp in self.lookupTable.index:
            expVals, _ = ScoreBoard.stat_cat_score(dice)
            dfTmp = pd.DataFrame(expVals.reshape(1,-1),
                                 columns=self.lutCols,
                                 index=[comp])
            self.lookupTable = self.lookupTable.append(dfTmp)
#        else:
#            lp('found in table')
#            lp(comp, self.lookupTable.loc[comp, :].values)
#            assert False
            
        res = self.lookupTable.loc[comp, :].values
#        lp(self.lookupTable)
#        lp(res, res.shape)
        return res
        

class PlayerEnsemble:
    """Representing a set of players with a set of probablilities to
    choose a specific one.
    Players with nGames==0 are never chosesn"""
    
    def __init__(self, players):
        """
        players : list of (weight : int, player : Player)
            weight is the probablility, no need to normalize
        """
        self.players = [pl[1] for pl in players]
        self.weights = [pl[0] for pl in players]
        
#        wSum = np.sum(self.weights)
#        self.weights = [w/wSum for w in self.weights]
#        if len(self.weights)>1:
#            self.weights[-1] = 1-np.sum(self.weights[:-1])
#        lp(self.weights, np.sum(self.weights))
#        probsSum = np.sum(
#        self.probs = [pr/np.sum(
    
    def rand(self):
        assert False, 'not yet needed, probably working'
#        lp(self.players)
#        lp(self.probs)
#        return random.choices(['a','b'], weights=[1,2])
#        lp(self.weights)
        flt = [hasattr(pl, 'nGames') and pl.nGames>0 for pl in self.players]
#        lp(flt)
        ps = [p for p,f in zip(self.players, flt) if f]
        ws = [w for w,f in zip(self.weights, flt) if f]
#        lp(ws)
        return random.choices(ps, ws)[0]
    
    def randGameSet(self):
#        lp(self.weights)
        flt = [not hasattr(pl, 'nGames') or
               (hasattr(pl, 'nGames') and pl.nGames>0)
               for pl in self.players]
        assert np.sum(flt) >= 1, (
                'no ready players in ensemble\n{:}\n{:}\n{:}'.format(
                        str(self.players), str(self.weights), str(flt)))
#        lp(flt)
        ps = [p for p,f in zip(self.players, flt) if f]
        ws = [w for w,f in zip(self.weights, flt) if f]
#        ps = self.players[flt]
#        ws = self.weights[flt]
#        lp(ws)
#        assert False
        
#        return random.choices(ps, ws, k=13*3)
        wSum = np.sum(ws)
        ws = [w/wSum for w in ws]
        return list(np.random.choice(ps, size=13*3, p=ws))

class AbstractPlayer(ABC):  # abstract class
 
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @abstractmethod
    def choose_reroll(self, scoreBoard, dice, attempt):
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
        list : boolean, len 5
            True: reroll this dice
        info : str
            Some information about the decision making
        
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
    def choose_cat(self, scoreBoard, dice):
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
        info : str
            Some information about the decision making
        
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
    
    def benchmark(self, nGames=100, nBins=20, seed=None):
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
        nBins : int
            before mean and std are calulated, the games scores are binned
            to nBins bins.
        
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
        if seed is not None:
            np.random.seed(seed)
        scores = []
        for ii in range(nGames):
            game = Game(self)
            scores += [game.log.loc[13, 'scoreBoard'].getSum()]
#        lp(scores)
        scores = [np.mean(chunk) for chunk in np.array_split(scores, nBins)]
#        lp(scores)
        return np.mean(scores), np.std(scores) #/len(scores)**.5
    
    @classmethod
    def modelBenchmark(cls, nGames=range(1,2), nInstances=50, *args, **kwargs):
        if not hasattr(cls, 'train'):
            nGames = [1]
        
        df = pd.DataFrame(index=nGames)
        df.index.name = 'nGames'
        
        for ii in progressbar(range(nInstances), 'Instances'):
            np.random.seed(ii)
            player = cls(*args, **kwargs)
            means = []
            stds = []
            for gg in nGames:
#                lp('game', gg)
                assert gg > 0, 'number of games <= 0 makes no sense'
                if hasattr(cls, 'train'):
                    player.train(nGames=gg-player.nGames)
                m, s = player.benchmark(nGames=50, nBins=10)
                means += [m]
                stds += [s]
            df['inst_'+str(ii)] = means

        res = pd.DataFrame()
        res['mean'] = df.mean(axis=1)
        res['sem'] = (df.std(axis=1)**2 + np.array(stds)**2)**.5 / nInstances**.5
        res['max'] = df.max(axis=1)
        res['min'] = df.min(axis=1)
        return res #, df

class PlayerTemporary(AbstractPlayer):
    """This is initiated with a choose and a category function
    and can be used for training"""
    name = 'temporaryPlayer'
    def __init__(self, fct_choose_cat, fct_choose_reroll):
        super().__init__()
        self.fct_choose_reroll = fct_choose_reroll
        self.fct_choose_cat = fct_choose_cat
    
    def choose_reroll(self, scoreBoard, dice, attempt):
        return self.fct_choose_reroll(scoreBoard, dice, attempt)
    
    def choose_cat(self, scoreBoard, dice):
        return self.fct_choose_cat(scoreBoard, dice)
    
class PlayerRandom(AbstractPlayer):
    """This player behaves completely random"""
    name = 'Random Player'
    def choose_reroll(self, scoreBoard, dice, attempt):
        return np.random.choice([True, False], 5), 'random choice'
    def choose_cat(self, scoreBoard, dice):
        return np.random.choice(scoreBoard.open_cats()), 'random choice'

class PlayerOneShotHero(AbstractPlayer):
    """This player assigns the dice to the category with the most scores"""
    name = 'Mr. One Shot Hero'
    def choose_reroll(self, scoreBoard, dice, attempt):
        return [False]*5
    def choose_cat(self, scoreBoard, dice):
#        print(151, dice, type(dice))
        bench = []
        for cat in scoreBoard.open_cats():
            score, bonus = scoreBoard.check_points(dice, cat)
            bench += [(score + bonus, cat)]
#        lp(bench)
        bench = sorted(bench, key=lambda x: x[0])
        return bench[-1][1]

class Player1ShotHuman(AbstractPlayer):
    """This player assigns the dice to the category with the most scores"""
    name = 'One Shot Human'
    def choose_reroll(self, scoreBoard, dice, attempt):
        return [False]*5
    def choose_cat(self, scoreBoard, dice):
        openCats = scoreBoard.open_cats()
        if len(openCats) == 1:
            return openCats[0]
        
        for cat in [11, 10, 9, 8, 7, 6]:
            score, bonus = scoreBoard.check_points(dice, cat)
            if score > 0 and cat in openCats:
                return cat
        
        # now we are left with multiples and chance
        hist = np.histogram(dice.vals, bins=np.linspace(.5,6.5,7))[0]
        if np.amax(hist) >= 3:
            if np.argmax(hist) in openCats:
                return np.argmax(hist)
        if np.amax(hist) == 2:
            argsMax = [ii for ii in range(6) if hist[ii]==2]
            argsMax = set(argsMax).intersection(set(openCats))
            argsMax = sorted(list(argsMax))
            if len(argsMax)>0:
                return argsMax[-1]
        if 12 in openCats and np.sum(dice.vals) >= 5*3.5:
            return 12 #chance
        # write 1 hit intocategories aces-fours
        for ii in range(6):
            if ii in openCats and ii in list(dice.vals):
                return ii
        # cancel categories
        for cat in [11, 10, 7, 8, 6, 9]:
            if cat in openCats:
                return cat
#        if 12 in openCats:
#            return 12 #chance
        
#        lp(openCats, dice)
        return openCats[0]





class PlayerOneShotAI(AbstractPlayer):
    """No strategic dice reroll, but self learning category assignment"""
    name = 'The One Shot AI'

    def __init__(
            self,
            hidden_layer_sizes=(40, 50, 40, 25, 20, 10),
#            hidden_layer_sizes=(40, 40, 25, 20, 10),
#            hidden_layer_sizes=(30, 25, 20),
#            regressor=MLPRegressor(hidden_layer_sizes=(30, 25, 20)),
#            debugLevel=0,
            catMLParas={'lenReplayMem': 200, 'lenMiniBatch': 30, 'gamma': .95}
#            catReplayMemLen=1000,
#            catMiniBatchSize=10,
            
#            playerInit=PlayerRandom(),
#            nGamesInit=1
            ):
        super().__init__()
        self.rgr = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
        self.nGames = 0
#        self.debugLevel = debugLevel
        self.catMLParas = catMLParas
        self.crm = []  # category replay memory
#        self.miniBatchSize = miniBatchSize
        
        # init regressor for adaptative fit
#        games = [Game(playerInit) for ii in range(nGamesInit)]
#        X, y = self.cat_decision_parser(*self.games_to_cat_info(games))
##        self.rgr.fit(X, y)
#        self.rgr.partial_fit(X, y)
        
    def choose_reroll(self, scoreBoard, dice, attempt):
        return [False]*5
    
    # move to abstract LearningPlayer
    def choose_cat(self, scoreBoard, dice, debugLevel=0):
        """Takes only open categories."""
        assert self.nGames > 0, str(scoreBoard.print())
#        if self.debugLevel > 0:
#            print('='*60)
#            print('='*60)
#            scoreBoard.print()
#            print('DICE: ', dice)
            
#        opts = []
#        for cat in scoreBoard.open_cats():
#            score = self.cat_predict(scoreBoard, dice, cat)
#            assert len(score)==1
#            score = score[0]
#            opts += [(cat, score)]
##            if self.debugLevel > 0:
##                print('cat: {:}, score: {:.2f}'.format(
##                        ScoreBoard.cats[cat], score))
##                print(323, ScoreBoard.cats[cat], score)
#        opts = sorted(opts, key=lambda x: x[1])
##        if self.debugLevel > 0:
##            print('-> chose cat:',ScoreBoard.cats[opts[-1][0]])
        
        opts = self.eval_cats(scoreBoard, dice)
        
        if debugLevel >0:
            info = '; '.join([
                    '{:}: {:.2f}'.format(ScoreBoard.cats[cat], score)
                    for cat, score in opts])
            lp(info)
        return opts[0][0]
#        cs = self.cat_predict(scoreBoard, dice)
##        cs = self.rgr.predict(self.cat_decision_parser(scoreBoard, dice))
#        cs = np.ma.masked_array(cs, mask=np.invert(scoreBoard.mask))
#        return np.argmax(cs)
    
    def predict_score(self, scoreBoard):
        """Predicts final (rest) score of a game based on the
        current score board.
        For now this is a rough approximation.
        ToDo: Use some machine learning model here
        """
        
        if False:  # FOR DEVELOPMENT ONLY: calc weights
            sb = ScoreBoard()
            weights = np.empty(shape=(13))
            for cc in range(13):
                res = []
                for ii in range(6**5):
                    dice = Dice()
                    score, bonus = sb.check_points(dice, cc)
                    res += [score]
                weights[cc] = np.mean(res)
            print(weights)
        
        ocs = scoreBoard.open_cats()
        expVals = np.array(
                [0.82317387, 1.68981481, 2.45679012, 3.32098765, 4.20203189,
                 5.04475309, 3.7188786, 0.34465021, 0.9837963, 4.49459877,
                 1.15740741, 0.02572016, 17.43454218])
        restScore = np.sum(expVals[ocs]) #*len(ocs)/2
        
#        print(scoreBoard)
#        lp(restScore)
        
        return restScore
    
    def eval_cats(self, scoreBoard, dice):
        """retuns list of (cat nr, predicted score)
        First element has highest score"""
        opts = []
        for cat in scoreBoard.open_cats():
            score = self.cat_predict(scoreBoard, dice, cat)
            assert len(score)==1
            score = score[0]
            
            # additionally consider the resulting state of the score board
            tmpSB = scoreBoard.copy()
            tmpSB.add(dice, cat)
            score += self.catMLParas['gamma'] * self.predict_score(tmpSB)
            
            
            opts += [(cat, score)]
#        lp(opts)
        opts = sorted(opts, key=lambda x: x[1], reverse=True)
#        lp(opts)
#        assert False
        return opts
    
    def cat_predict(self, scoreBoard, dice, cat):
#        X, y = self.cat_decision_parser(scoreBoard, dice, cat)
#        lp(X)
        x = self.encoder(scoreBoard, dice, cat).reshape(1, -1)
        return self.rgr.predict(x)
    
    def games_to_cat_replay_memory(self, games):
        games = list_cast(games)
        for game in games:
            for rr in range(12):
                sb1, dice1, cat1 = game.catLog[rr]
                sb2, dice2, cat2 = game.catLog[rr+1]
                sc = sb2.getSum() - sb1.getSum()
                self.crm += [(sb1, dice1, cat1, sc, sb2, dice2)]
                
            sb1, dice1, deci1 = game.catLog[12]
            sc = game.sb.getSum() - sb1.getSum()
            self.crm += [(sb1, dice1, cat1, sc)]
            
        
        
    def cat_replay_memory_trunc(self):
#        self.crm = sorted(self.crm, key=lambda x: x[3])  # sort by score
        self.crm = self.crm[-self.catMLParas['lenReplayMem']:]
    
    def xy_from_crm(self, crmElem):
        assert len(crmElem) in [4,6]
        sb1, dice1, cat1, sc = crmElem[:4]
        
        x = self.encoder(sb1, dice1, cat1)
        y = sc
        if len(crmElem) == 6 and self.nGames > 0:
            sb2, dice2 = crmElem[4:]
            
            # forecast rest of the game with the same regressor
            # bad because of dice2
#            opts = self.eval_cats(sb2, dice2)
#            y += self.catMLParas['gamma'] * opts[0][1]
            
            y += self.catMLParas['gamma'] * self.predict_score(sb2)


            
        return x, y
    
#    def games_to_cat_info(self, games):
#        """Extracts relevant information for category prediction
#        from the games log
#        """
#        games = list_cast(games)
#        scoreBoards = [game.log[rr][0] for game in games for rr in range(13)]
#        dices = [game.log[rr][-2] for game in games for rr in range(13)]
#        cats = [game.log[rr][-1] for game in games for rr in range(13)]
#        return scoreBoards, dices, cats
#    
#    def cat_decision_parser(self, scoreBoards, dices, cats):
#        """Prepares X, y tupes for regressor fit
#        based on relevant information from games_to_cat_info
#        cat : int
#        """
#        scoreBoards = list_cast(scoreBoards)
#        lp(dices)
#        dices = list_cast(dices)
#        cats = list_cast(cats)
#        assert len(scoreBoards) == len(dices) == len(cats)
#        n_samples = len(scoreBoards)
##        lp(n_samples, self.n_features)
##        a = np.empty(shape=(13,31))
#        X = np.empty(shape=(n_samples, self.n_features))
#        y = np.empty(shape=(n_samples,))
#        for ind in range(n_samples):
##        for ind, (scoreBoard, dice) in enumerate(zip(scoreBoards, dices)):
##            gs = game.score
##            for rr, roundLog in enumerate(games.log):
##                ind = gg*13 + rr
##            scoreBoard.print()
#            scoreBoard = scoreBoards[ind]
#            dice = dices[ind]
#            cat = cats[ind]
#            lp(dice, type(dice))
#            X[ind, :] = self.encoder(scoreBoard, dice, cat)
#            y[ind] = scoreBoard.score
#        return X, y
    
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13
    def encoder(self, scoreBoard, dice, cat):
        """Encodes a game situation (decision input) as
        array with elements in range 0 to 1.
        """
        x = np.zeros(shape=(self.n_features))
        score, bonus = scoreBoard.check_points(dice, cat)
        x[cat] = score
##        lp(scoreBoard.scores.data)
#        x[:13] = scoreBoard.scores.data / 50  # scores
#        x[13:26] = scoreBoard.scores.mask.astype(int)  # avail. cats
#        x[26:31] = (dice.vals -1) / 5  # dice
#        x[31] = cat
        return x
        # hopefully learns bonus by itsself
#        x[0, 31] = np.sum(roundLog[0].data[:6]) / 105  # upper sum for bonus
    
    def train_dep(self, nGames,
              trainerEnsemble=PlayerEnsemble(
                      [(1, PlayerRandom())]
                      )):
        """Training the Player with nGames and based on the trainers moves.
    
        Extended description of function.
    
        Parameters
        ----------
        nGames : int
            Nomber of games
        trainerEnsemble : PlayerEnsemble
            Integer represents the weight of the specici players moves
            player is someone doing decisions; None is self
    
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
        """use:
            MLPRegressor
            partial_fit
            https://www.programcreek.com/python/example/93778/sklearn.neural_network.MLPRegressor
        """
#        plys = [tr[1] for tr in trainers]
#        probs = [tr[0] for tr in trainers]
        
        for gg in range(nGames):
#            player =A np.random.choice(plys, p=probs)
            players = trainerEnsemble.randGameSet()
#            lp(type(player), len(player))
#            if player is None:
#                player = self
#            lp(type(player), len(player))
            if hasattr(players[0], 'nGames'):
                if players[0].nGames <= 0:
                    players[0] = PlayerRandom()
#            lp(type(players), len(players))
            game = Game(players)
            
            self.games_to_cat_replay_memory(game)
            #create miniBatch
            n_samples = self.catMLParas['lenMiniBatch']
            X = np.empty(shape=(n_samples, self.n_features))
            y = np.empty(shape=(n_samples,))
            for ind in range(n_samples):
                crmElem = np.random.choice(self.crm)
#                lp(crmElem[1].vals, ScoreBoard.cats[crmElem[2]], crmElem[3])
                xy = self.xy_from_crm(crmElem)
                X[ind, :] = xy[0]
                y[ind] = xy[1]
#                lp(X[ind,:])
#                lp(y[ind])
#            scoreBoards, dices, cats = self.games_to_cat_info(game)
#            lp(scoreBoards)
#            lp(dices)
#            lp(cats)
#            X, y = self.cat_decision_parser(scoreBoards, dices, cats)

            self.rgr.partial_fit(X, y)
            self.nGames += 1
            
    def train(self, nGames, pRand=.1, pRat=100):
        """Training the Player with nGames and based on the trainers moves.
    
        Extended description of function.
    
        Parameters
        ----------
        nGames : int
            Nomber of games
        trainerEnsemble : PlayerEnsemble
            Integer represents the weight of the specici players moves
            player is someone doing decisions; None is self
        pRat : float
            predicted best action is pRat times as probable to choose as
            the predicted most unfavourable action.
    
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
        """use:
            MLPRegressor
            partial_fit
            https://www.programcreek.com/python/example/93778/sklearn.neural_network.MLPRegressor
        """
        
#        plys = [tr[1] for tr in trainers]
#        probs = [tr[0] for tr in trainers]
        
        for gg in range(nGames):
            game = Game()
            for rr in range(13):
                for aa in range(3):
                    act, paras = game.ask_action()
                    if aa < 2:
                        game.perf_action(act, self.choose_reroll(*paras))
                    else:
                        if self.nGames == 0 or np.random.rand() <= pRand:
                            cat = np.random.choice(paras[0].open_cats())
                        else:
#                            cat = self.choose_cat(*paras)
                            opts = self.eval_cats(*paras)
                            
                            cs = [opt[0] for opt in opts]
#                            lp(cs)
                            ws = [opt[1] for opt in opts]
#                            lp(ws)
                            ws = np.array(ws) - np.amin(ws)
#                            lp(ws)
                            if np.amax(ws) > 0:
                                alpha = np.log(pRat)/np.amax(ws)
                                ws = np.exp(alpha*ws)
#                                lp(ws)
                                cat = weighted_choice(cs, ws)
                            else:
                                cat = opts[0][0]
#                            assert False
                        game.perf_action(act, cat)
#            assert False
#            player =A np.random.choice(plys, p=probs)
#            players = trainerEnsemble.randGameSet()
#            lp(type(player), len(player))
#            if player is None:
#                player = self
#            lp(type(player), len(player))
#            if hasattr(self, 'nGames'):
#                if players[0].nGames <= 0:
#                    players[0] = PlayerRandom()
#            players = 0
#            if self.nGames == 0:
#                player = PlayerRandom()
##            lp(type(players), len(players))
#            game = Game(players)
            
            self.games_to_cat_replay_memory(game)
            #create miniBatch
            n_samples = self.catMLParas['lenMiniBatch']
            X = np.empty(shape=(n_samples, self.n_features))
            y = np.empty(shape=(n_samples,))
            for ind in range(n_samples):
                crmElem = np.random.choice(self.crm)
#                if self.nGames == 0:
#                    crmElem = self.crm[-1]
#                lp(crmElem[1].vals, ScoreBoard.cats[crmElem[2]], crmElem[3])
#                lp('Dice:', crmElem[1].vals)
#                lp('Chosen cat:', ScoreBoard.cats[crmElem[2]], '; n open calts:', len(crmElem[0].open_cats()))
#                lp('Result Score:', crmElem[3])
                xy = self.xy_from_crm(crmElem)
                X[ind, :] = xy[0]
                y[ind] = xy[1]
#                lp('X', X[ind,:])
#                lp('y', y[ind])
#            scoreBoards, dices, cats = self.games_to_cat_info(game)
#            lp(scoreBoards)
#            lp(dices)
#            lp(cats)
#            X, y = self.cat_decision_parser(scoreBoards, dices, cats)

#            self.rgr.partial_fit(X, y)  # TODO self.scrRgr = self.scrRgr.partial_fit(X, y)
            self.rgr = self.rgr.partial_fit(X, y)
            self.cat_replay_memory_trunc()
            self.nGames += 1
    
#    @classmethod
#    def modelBenchmark(cls, nGames=range(1,2), nInstances=50, *args, **kwargs):
#        np.random.seed(0)
#        df = pd.DataFrame(index=nGames)
#        df.index.name = 'nGames'
##        print('inst', flush=True, end='; ')
#        for ii in progressbar(range(nInstances), 'Instances'):
##            print(ii, flush=True, end='; ')
#            player = cls(*args, **kwargs)
#            means = []
#            stds = []
#            for gg in nGames:
##                lp('game', gg)
#                assert gg > 0, 'number of games <= 0 makes no sense'
#                player.train2(nGames=gg-player.nGames)
#                m, s = player.benchmark(nGames=50, nBins=10)
#                means += [m]
#                stds += [s]
#            df['inst_'+str(ii)] = means
##        print()
#        res = pd.DataFrame()
#        res['mean'] = df.mean(axis=1)
#        res['sem'] = (df.std(axis=1)**2 + np.array(stds)**2)**.5 / nInstances**.5
#        res['max'] = df.max(axis=1)
#        res['min'] = df.min(axis=1)
#        return res #, df



class PlayerAI_1SEnc_1(PlayerOneShotAI):
    name = 'AI 1S Enc 1Score13Scores'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
#        lp(self.rgr.get_params())
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 1 + 13
    def encoder(self, scoreBoard, dice, cat):
        """Encoding:
        0: score in cat
        1-13: scores in all cats
        """
        assert 0 <= cat <= 12
        x = np.zeros(shape=(self.n_features))
        score, bonus = scoreBoard.check_points(dice, cat)
        x[0] = score
        
        for cc in scoreBoard.open_cats():
            if cc==cat:
                continue  # helps converging
            score, bonus = scoreBoard.check_points(dice, cc)
            x[1+cc] = score
        return x




class PlayerAI_1SEnc_2(PlayerOneShotAI):
    name = 'AI 1S Enc 2Sore13RelativeScores'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 2 + 13
    def encoder(self, scoreBoard, dice, cat):
        """Encoding:
        0: score in cat
        1: relative score in cat
        2-14: all relative scores
        """
        assert 0 <= cat <= 12
        x = np.zeros(shape=(self.n_features))
        score, bonus = scoreBoard.check_points(dice, cat)
        x[0] = score
        x[1] = score / scoreBoard.check_points_max(cat)
        
        for cc in scoreBoard.open_cats():
            if cc==cat:
                continue
            score, bonus = scoreBoard.check_points(dice, cc)
            x[2+cc] = score / scoreBoard.check_points_max(cc)
        return x
    
    



class PlayerAI_1SEnc_3(PlayerOneShotAI):
    name = 'AI 1S Enc 1Score'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 1
    def encoder(self, scoreBoard, dice, cat):
        """Encoding:
        0: score in cat
        """
        assert 0 <= cat <= 12
        x = np.zeros(shape=(self.n_features))
        score, bonus = scoreBoard.check_points(dice, cat)
        x[0] = score
        return x

class PlayerAI_1SEnc_4(PlayerOneShotAI):
    name = 'AI 1S Enc 13Score13cats6Dice'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 13 +6
    def encoder(self, scoreBoard, dice, cat):
        """Encoding:
        13: cat flag
        13: open cats without cat
        6: dice hist
        """
        assert 0 <= cat <= 12
        x = np.zeros(shape=(self.n_features))
        score, bonus = scoreBoard.check_points(dice, cat)
        x[cat] = score
        for cc in scoreBoard.open_cats():
            x[13+cc] = 1
        x[26:32] = np.histogram(dice.vals, bins=np.linspace(.5,6.5,7))[0]
        return x

class PlayerAI_1SEnc_5(PlayerOneShotAI):
    name = 'AI 1S Enc 13Score13Scores'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
#        lp(self.rgr.get_params())
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 13
    def encoder(self, scoreBoard, dice, cat):
        """Encoding:
        13: score in cat
        13: scores in all cats
        """
        assert 0 <= cat <= 12
        x = np.zeros(shape=(self.n_features))
        score, bonus = scoreBoard.check_points(dice, cat)
        x[cat] = score
        
        for cc in scoreBoard.open_cats():
            if cc==cat:
                continue  # helps converging
            score, bonus = scoreBoard.check_points(dice, cc)
            x[13+cc] = score
        return x

class PlayerAI_1SEnc_6(PlayerOneShotAI):
    name = 'AI 1S Enc 13Score'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
#        lp(self.rgr.get_params())
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13
    def encoder(self, scoreBoard, dice, cat):
        """Encoding:
        13: score in cat
        """
        assert 0 <= cat <= 12
        x = np.zeros(shape=(self.n_features))
        score, bonus = scoreBoard.check_points(dice, cat)
        x[cat] = score

        return x


            
    
class PlayerOneShotAI_new(AbstractPlayer):
    """New AI Player concept.
    
    Regressor scrRgr focasts the rest score of the game
    based on the empty categories of a score board.
    
    Categorie decision is made based on the direct reward 
    + the reward forcast of the resulting scoreboard.
    """
    name = 'AI 1S scrRgr'
    def __init__(
            self, mlpRgrArgs={'hidden_layer_sizes':(20, 10)},
            lenScrReplayMem=13*100, lenMiniBatch=3000, gamma=1):
        """
        mlpRgrArgs : dict
            Arguments passed to MLPRegressor
        lenScrReplayMem : int
            Length of the replay memory for self.scrRgr training
        lenMiniBatch : int
            Number of samples used for each training iteration
        gamma : 0 <= gamma <= 1
            Damping factor for future rewards
        """
        super().__init__()
        
        self.scrRgr = MLPRegressor(**mlpRgrArgs)
#        self.scrRgr = MLPRegressor(hidden_layer_sizes=(40, 50, 40, 25, 20, 10))
        self.srm = []
        self.lenScrReplayMem = lenScrReplayMem
        self.lenMiniBatch = lenMiniBatch
        self.gamma = gamma
        self.nGames = 0
#        lp(self.scrRgr.get_params())
#        assert False
#        lp('todo check gamma=0')
        
        
    def choose_reroll(self, scoreBoard, dice, attempt):
        return [False]*5
    
    def choose_cat(self, scoreBoard, dice, debugLevel=0):
        opts = self.eval_options_cat(scoreBoard, dice)
        return opts[0][0]
    
    def eval_options_cat(self, scoreBoard, dice, debug=0):
        """Return a sorted list with the options to choose for cat and
        the expected restScore.
        """
        opts = []
        if debug==1:
            lp('todo: check reward', self.gamma)
            lp(scoreBoard, dice)
        for cat in scoreBoard.open_cats():
            directReward, bonus = scoreBoard.check_points(dice, cat)
#            score = self.cat_predict(scoreBoard, dice, cat)
#            assert len(score)==1
#            score = score[0]
            
            # additionally consider the resulting state of the score board
            tmpSB = scoreBoard.copy()
            tmpSB.add(dice, cat)
#            score += self.catMLParas['gamma'] * self.predict_score(tmpSB)
            x = self.encode_scrRgr_x(tmpSB).reshape(1, -1)
            futureReward = self.scrRgr.predict(x)[0]
#            lp(x, futureReward)
            
            reward = directReward + self.gamma * futureReward
            opts += [(cat, reward, directReward, futureReward)]
            
            if debug==1:
                lp(ScoreBoard.cats[cat], directReward, futureReward)
            
        opts = sorted(opts, key=lambda x: x[1], reverse=True)
        if debug==1:
            lp(opts)
        return opts
    
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13
    def encode_scrRgr_x(self, scoreBoard):
        """Encodes a scoreboard to a numpy array,
        which is used as the scrRgr input layer
        """
        x = np.zeros(shape=(self.n_features))
        x[:13] = scoreBoard.mask.astype(int)
        # todo: later add here upper sum for bonus consideration
#        lp('todo check encoding')
#        lp(scoreBoard)
#        lp(x)
        return x
    
    def add_srm_sample(self, scoreBoard, restScore):
        """Save memory in format x, y"""
        x = self.encode_scrRgr_x(scoreBoard)
        y = restScore
#        lp(x, y)
        self.srm += [(x, y)]
#        lp(self.srm)
    
    def truncate_srm(self):
        """Reduce srm length to lenScrReplayMem to the most recent memories.
        """
        self.srm = self.srm[-self.lenScrReplayMem:]
    
    def train(self, nGames, pRand=0.1, pRat=100):
        """Training the Player with nGames and based on the trainers moves.
    
        Extended description of function.
    
        Parameters
        ----------
        nGames : int
            Nomber of games
        trainerEnsemble : PlayerEnsemble
            Integer represents the weight of the specici players moves
            player is someone doing decisions; None is self
        pRat : float
            predicted best action is pRat times as probable to choose as
            the predicted most unfavourable action.
            None: switch of and the the best option
    
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
        for gg in range(nGames):
            game = Game()
            sbs = []
            for rr in range(13):
                sbs += [game.sb.copy()]  # only for every round
                for aa in range(3):
                    act, paras = game.ask_action()
                    if aa < 2:
                        sb, dice, attempt = paras
                        game.perf_action(act,
                                         self.choose_reroll(sb, dice, attempt))
                    else:
                        sb, dice = paras
                        if self.nGames == 0 or np.random.rand() < pRand:
#                            if self.nGames > 0:
#                                assert False
                            cat = np.random.choice(sb.open_cats())
                            
                        elif pRat is None:
                            cat = self.choose_cat(sb, dice)
                            
                            # for debugging
#                            lp('choose cat:', ScoreBoard.cats[cat])
#                            opts = self.eval_options_cat(sb, dice, debug=0)
#                            lp(opts)
                        else:
#                            if self.nGames > 0:
#                                assert False
                            opts = self.eval_options_cat(sb, dice)
                            
                            # chose an option for training which promisses a
                            # high score. A weighted choose is performed where
                            # the best option is pRat times as likely as
                            # the worst option
                            cs = [opt[0] for opt in opts]
                            ws = [opt[1] for opt in opts]
                            ws = np.array(ws) - np.amin(ws)
                            if np.amax(ws) > 0:
                                alpha = np.log(pRat)/np.amax(ws)
                                ws = np.exp(alpha*ws)
                                cat = weighted_choice(cs, ws)
                            else:
                                cat = opts[0][0]
                        game.perf_action(act, cat)
#                    lp(rr, aa, len(sbs))
            finalScore = game.sb.getSum()
            
            
            for sb in sbs:
#                lp(sb)
#                lp(finalScore, sb.getSum(), finalScore-sb.getSum())
                self.add_srm_sample(sb, finalScore-sb.getSum())
            self.truncate_srm()
            
#            if True:
            if self.nGames ==0:
                n_samples = len(self.srm)
                X = np.empty(shape=(n_samples, self.n_features))
                y = np.empty(shape=(n_samples,))
                for nn in range(n_samples):
                    xy = self.srm[nn]
                    X[nn, :] = xy[0]
                    y[nn] = xy[1]
#                self.scrRgr = self.scrRgr.fit(X, y)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    self.scrRgr = self.scrRgr.fit(X, y)
            else:
            
                #partial fit by miniBatch
                n_samples = self.lenMiniBatch
                X = np.empty(shape=(n_samples, self.n_features))
                y = np.empty(shape=(n_samples,))
                for nn in range(n_samples):
                    ind = np.random.choice(list(range(len(self.srm))))
                    xy = self.srm[ind]
                    X[nn, :] = xy[0]
                    y[nn] = xy[1]
                self.scrRgr = self.scrRgr.partial_fit(X, y)

#            for nn in range(n_samples):
#                lp(nn, X[nn:nn+1, :], y[nn], self.scrRgr.predict(X[nn:nn+1, :]))

            self.nGames += 1




class PlayerOneShotAI_v2(AbstractPlayer):
    """New AI Player concept.
    
    Regressor scrRgr focasts the rest score of the game
    based on the empty categories of a score board.
    
    Categorie decision is made based on the direct reward 
    + the reward forcast of the resulting scoreboard.
    """
    name = 'AI 1S scrRgr'
    def __init__(
            self, mlpRgrArgs={'hidden_layer_sizes':(20, 10)},
            lenScrReplayMem=13*10, lenMiniBatch=3000, gamma=1):
        """
        mlpRgrArgs : dict
            Arguments passed to MLPRegressor
        lenScrReplayMem : int
            Length of the replay memory for self.scrRgr training
        lenMiniBatch : int
            Number of samples used for each training iteration
        gamma : 0 <= gamma <= 1
            Damping factor for future rewards
        """
        super().__init__()
        
        self.scrRgr = MLPRegressor(**mlpRgrArgs)
#        self.scrRgr = MLPRegressor(hidden_layer_sizes=(40, 50, 40, 25, 20, 10))
        self.srm = []
        self.lenScrReplayMem = lenScrReplayMem
        self.lenMiniBatch = lenMiniBatch
        self.gamma = gamma
        self.nGames = 0
#        lp(self.scrRgr.get_params())
#        assert False
#        lp('todo check gamma=0')
        
        
    def choose_reroll(self, scoreBoard, dice, attempt):
        return [False]*5
    
    def choose_cat(self, scoreBoard, dice, debugLevel=0):
        opts = self.eval_options_cat(scoreBoard, dice)
        return opts[0][0]
    
    def eval_options_cat(self, scoreBoard, dice, debug=0):
        """Return a sorted list with the options to choose for cat and
        the expected restScore.
        """
        opts = []
        if debug==1:
            lp('todo: check reward', self.gamma)
            lp(scoreBoard, dice)
        for cat in scoreBoard.open_cats():
            directReward, bonus = scoreBoard.check_points(dice, cat)
#            score = self.cat_predict(scoreBoard, dice, cat)
#            assert len(score)==1
#            score = score[0]
            
            # additionally consider the resulting state of the score board
            tmpSB = scoreBoard.copy()
            tmpSB.add(dice, cat)
#            score += self.catMLParas['gamma'] * self.predict_score(tmpSB)
            x = self.encode_scrRgr_x(tmpSB).reshape(1, -1)
            futureReward = self.scrRgr.predict(x)[0]
#            lp(x, futureReward)
            
            reward = directReward + self.gamma * futureReward
            opts += [(cat, reward, directReward, futureReward)]
            
            if debug==1:
                lp(ScoreBoard.cats[cat], directReward, futureReward)
            
        opts = sorted(opts, key=lambda x: x[1], reverse=True)
        if debug==1:
            lp(opts)
        return opts
    
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13
    def encode_scrRgr_x(self, scoreBoard):
        """Encodes a scoreboard to a numpy array,
        which is used as the scrRgr input layer
        """
        x = np.zeros(shape=(self.n_features))
        x[:13] = scoreBoard.mask.astype(int)
        # todo: later add here upper sum for bonus consideration
#        lp('todo check encoding')
#        lp(scoreBoard)
#        lp(x)
        return x
    
    def add_sbs_to_srm(self, sbs):
        """Add expereience to score regressor memory
        sb : ScoreBoard
        reward int
        """
        for ii in range(len(sbs)-1):
            sb1 = sbs[ii]
            sb2 = sbs[ii+1]
            reward = sb2.getSum() - sb1.getSum()
            self.srm += [(sb1, reward, sb2)]
        self.truncate_srm()

    
    def truncate_srm(self):
        """Reduce srm length to lenScrReplayMem to the most recent memories.
        """
        self.srm = self.srm[-self.lenScrReplayMem:]
    
    def train(self, nGames, pRand=0.1, pRat=100):
        """Training the Player with nGames and based on the trainers moves.
    
        Extended description of function.
    
        Parameters
        ----------
        nGames : int
            Nomber of games
        trainerEnsemble : PlayerEnsemble
            Integer represents the weight of the specici players moves
            player is someone doing decisions; None is self
        pRat : float
            predicted best action is pRat times as probable to choose as
            the predicted most unfavourable action.
            None: switch of and the the best option
    
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
        for gg in range(nGames):
            game = Game()
            sbs = []
#            #initial training round
#            if self.nGames == 0:
#                for rr in range(13):
#                    sbs += [game.sb.copy()]  # only for every round
#                    for aa in range(3):
#                        act, paras = game.ask_action()
#                        if aa < 2:
#                            sb, dice, attempt = paras
#                            game.perf_action(act,
#                                             self.choose_reroll(sb, dice, attempt))
#                        else:
#                            sb, dice = paras
#                            cat = np.random.choice(sb.open_cats())
#                            game.perf_action(act, cat)
#                sbs += [game.sb.copy()]
#                self.add_sbs_to_srm(sbs)
#                continue
                            
                        
            for rr in range(13):
                sbs += [game.sb.copy()]  # only for every round
                for aa in range(3):
                    act, paras = game.ask_action()
                    if aa < 2:
                        sb, dice, attempt = paras
                        game.perf_action(act,
                                         self.choose_reroll(sb, dice, attempt))
                    else:
                        sb, dice = paras
                        if self.nGames == 0 or np.random.rand() < pRand:
                            cat = np.random.choice(sb.open_cats())            
                        elif pRat is None:
                            cat = self.choose_cat(sb, dice)
                        else:
                            opts = self.eval_options_cat(sb, dice)
                            
                            # chose an option for training which promisses a
                            # high score. A weighted choose is performed where
                            # the best option is pRat times as likely as
                            # the worst option
                            cs = [opt[0] for opt in opts]
                            ws = [opt[1] for opt in opts]
                            ws = np.array(ws) - np.amin(ws)
                            if np.amax(ws) > 0:
                                alpha = np.log(pRat)/np.amax(ws)
                                ws = np.exp(alpha*ws)
                                cat = weighted_choice(cs, ws)
                            else:
                                cat = opts[0][0]
                        game.perf_action(act, cat)
#                    lp(rr, aa, len(sbs))
            sbs += [game.sb.copy()]
            self.add_sbs_to_srm(sbs)
#            finalScore = game.sb.getSum()
            
            
#            for sb in sbs:
##                lp(sb)
##                lp(finalScore, sb.getSum(), finalScore-sb.getSum())
#                self.add_srm_sample(sb, finalScore-sb.getSum())
            
            
#            if True:
            if self.nGames ==0:
                n_samples = len(self.srm)
                X = np.empty(shape=(n_samples, self.n_features))
                y = np.empty(shape=(n_samples,))
                for nn in range(n_samples):
                    sb1, reward, sb2 = self.srm[nn]
                    X[nn, :] = self.encode_scrRgr_x(sb1).reshape(1,-1)
                    y[nn] = reward
#                self.scrRgr = self.scrRgr.fit(X, y)
#                lp(0)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    self.scrRgr = self.scrRgr.fit(X, y)
                
            else:
#                lp('>0')
#                assert False
                #partial fit by miniBatch
                n_samples = self.lenMiniBatch
                X = np.empty(shape=(n_samples, self.n_features))
                y = np.empty(shape=(n_samples,))
                for nn in range(n_samples):
                    ind = np.random.choice(list(range(len(self.srm))))
                    sb1, dirRew, sb2 = self.srm[ind]
                    X[nn, :] = self.encode_scrRgr_x(sb1).reshape(1,-1)
                    xSb2 = self.encode_scrRgr_x(sb2).reshape(1,-1)
                    futRew = self.scrRgr.predict(xSb2)[0]
#                     x = self.encode_scrRgr_x(tmpSB).reshape(1, -1)
#                     futureReward = self.scrRgr.predict(x)[0]
                    y[nn] = dirRew + self.gamma * futRew
#                    xy = self.srm[ind]
#                    X[nn, :] = xy[0]
#                    y[nn] = xy[1]
                self.scrRgr = self.scrRgr.partial_fit(X, y)

#            for nn in range(n_samples):
#                lp(nn, X[nn:nn+1, :], y[nn], self.scrRgr.predict(X[nn:nn+1, :]))

            self.nGames += 1


class PlayerAI_full_v0(AbstractPlayer):
    """AI Player concept: simple, re-roll implementation.
    
    Regressor scrRgr focasts the rest score of the game
    based on the empty categories of a score board.
    
    Categorie decision is made based on the direct reward 
    + the reward forcast of the resulting scoreboard.
    """
    name = 'PlayerAI_full_v0'
    def __init__(
            self,
            scrRgrArgs={'hidden_layer_sizes':(20, 10)},
            lenScrReplayMem=13*100, lenScrMiniBatch=13*50,
            rrRgrArgs={'hidden_layer_sizes':(40, 40, 40, 40, 40, 10)},
            lenRrReplayMem=26*100, lenRrMiniBatch=26*50,
            nIterPartFit=5,
            gamma=1,
            fn=None):
        """
        mlpRgrArgs : dict
            Arguments passed to MLPRegressor
        lenScrReplayMem : int
            Length of the replay memory for self.scrRgr training
        lenMiniBatch : int
            Number of samples used for each training iteration
        gamma : 0 <= gamma <= 1
            Damping factor for future rewards
        """
        super().__init__()
        
        self.scrRgr = MLPRegressor(**scrRgrArgs)
        self.rrRgr = MLPRegressor(**rrRgrArgs)
#        self.scrRgr = MLPRegressor(hidden_layer_sizes=(40, 50, 40, 25, 20, 10))
        self.srm = []
        self.rrm = []
        self.lenScrReplayMem = lenScrReplayMem
        self.lenScrMiniBatch = lenScrMiniBatch
        self.lenRrReplayMem = lenRrReplayMem
        self.lenRrMiniBatch = lenRrMiniBatch
        self.nIterPartFit = nIterPartFit
        self.gamma = gamma
        self.nGames = 0
        
        if fn is not None:
            self.load(fn)
#        lp(self.scrRgr.get_params())
#        assert False
#        lp('todo check gamma=0')
        
        
    def choose_reroll(self, scoreBoard, dice, attempt):
        opts = self.eval_options_reroll(scoreBoard, dice, attempt)
        return opts[0][0], str(opts)
    
    def choose_cat(self, scoreBoard, dice, debugLevel=0):
        opts = self.eval_options_cat(scoreBoard, dice)
        return opts[0][0], str(opts)
    
    def eval_options_cat(self, scoreBoard, dice, debug=0):
        """Return a sorted list with the options to choose for cat and
        the expected restScore.
        """
        opts = []
        if debug==1:
            lp('todo: check reward', self.gamma)
            lp(scoreBoard, dice)
        for cat in scoreBoard.open_cats():
            directReward, bonus = scoreBoard.check_points(dice, cat)
            uSum = scoreBoard.getUpperSum()
            if cat <= 5 and uSum < 63 and uSum + directReward >= 63:
                directReward += 35
#            score = self.cat_predict(scoreBoard, dice, cat)
#            assert len(score)==1
#            score = score[0]
            
            # additionally consider the resulting state of the score board
            tmpSB = scoreBoard.copy()
            tmpSB.add(dice, cat)
#            score += self.catMLParas['gamma'] * self.predict_score(tmpSB)
            x = self.encode_scrRgr_x(tmpSB).reshape(1, -1)
            futureReward = self.scrRgr.predict(x)[0]
#            lp(x, futureReward)
            
            reward = directReward + self.gamma * futureReward
            opts += [(cat, reward, directReward, futureReward)]
            
            if debug==1:
                lp(ScoreBoard.cats[cat], directReward, futureReward)
            
        opts = sorted(opts, key=lambda x: x[1], reverse=True)
        if debug==1:
            lp(opts)
        return opts
    
    def eval_options_reroll(self, sb, dice, att):
        opts = []
        keepDices = []
        for reroll in product([True, False], repeat=5):
            # avoid unnecessary rerolls ([1, 2r, 2, 3, 4] and [1, 2, 2r, 3, 4])
            keepDice = dice.vals[np.logical_not(reroll)]
#            if keepDice in keepDices:
            if arreq_in_list(keepDice, keepDices):
                continue
            else:
                keepDices += [keepDice]
#            lp(reroll)
            x = self.encode_rrRgr_x(sb, att, dice, reroll).reshape(1, -1)
            reward = self.rrRgr.predict(x)[0]
            opts += [(reroll, reward)]
#        lp(opts)
        opts = sorted(opts, key=lambda x: x[1], reverse=True)
        return opts
    
    @property
    def nFeat_scrRgr(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 1
    def encode_scrRgr_x(self, scoreBoard):
        """Encodes a scoreboard to a numpy array,
        which is used as the scrRgr input layer
        """
        x = np.zeros(shape=(self.nFeat_scrRgr))
        x[:13] = scoreBoard.mask.astype(int)
        x[13] = scoreBoard.getUpperSum() / 63
        # todo: later add here upper sum for bonus consideration
#        lp('todo check encoding')
#        lp(scoreBoard)
#        lp(x)
        return x
    
    @property
    def nFeat_rrRgr(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 1 + 6 + 1 + 5
    def encode_rrRgr_x(self, scoreBoard, attempt, dice, reroll):
        """Encodes a scoreboard to a numpy array,
        which is used as the scrRgr input layer
        13: cat available
        1: attempt
        6: fixed dice histogram
        1: upper sum score board for bonus
        7 helper quantities
        - 1: sum of dice
        - 1: small straight
        - 1: large straight
        - 2: number of double, number of tripple, ...
        """
        keepDice = dice.vals[np.logical_not(reroll)]
        x = np.zeros(shape=(self.nFeat_rrRgr))
        x[:13] = scoreBoard.mask.astype(int)
#        x[:13] = -1
#        for cc in scoreBoard.open_cats():
#            if len(keepDice) > 0:
#                x[cc] = scoreBoard.check_points(keepDice, cc)
#            else:
#                x[cc] = 0

        x[13] = attempt
        #how many dice are kept in categories 1-6
        
        hist = np.histogram(keepDice, bins=np.linspace(.5,6.5,7))[0]
        x[14:20] = hist * np.array([1,2,3,4,5,6])
#        for cc in scoreBoard.open_cats():
#            if cc >= 6:
#                break
#            x[14+cc] = hist[cc] * (cc+1)
        
        # Simplified Bonus indicator
        x[20] = scoreBoard.getUpperSum() / 63
#        upSum = scoreBoard.getUpperSum()
#        if upSum >= 63 or np.sum(scoreBoard.mask[:6])==0:
##            x[20] = (5*(2+3+4+5+6) - 63) / 5  # max posible values
#            x[20] = -63/5  # min possible value
#        else:
#            # nbp: needed bonus progress (i.e. 3 in each cat)
#            nbp = np.sum(
#                    np.logical_not(scoreBoard.mask[:6]).astype(int)
#                    * np.array(range(1,7)) * 3)
#            mrbp = (63 - nbp) / 3 * 5  # max reachable bonus points
#            x[20] = (upSum - nbp) / mrbp
##            lp(scoreBoard.scores[:6])
##            lp(nbp, scoreBoard.getUpperSum(), x[20])
#        # prevent impossible bonus attempts
#        if x[20] < -1:
#            x[20] = -63/5


        
        # Helper for 3/4 of a kind and chance
        x[21] = np.sum(keepDice)
        
        #elements for small straight
#        if x[9] == 1:
        for ii in range(3):
            x[22] = max(x[22], np.sum(hist[ii:ii+4]>0))
        #elements for large straight
#        if x[10] == 1:
        for ii in range(2):
            x[23] = max(x[23], np.sum(hist[ii:ii+5]>0))
#        #elements for full house
#        if x[8] == 1:
        # doble, tripple ,...
        x[24] = len(hist[hist==2]) # number of doubles
        x[25] = len(hist[hist==3]) # number of tripples
#        x[26] = len(hist[hist==4])
#        x[27] = len(hist[hist==5])
        
        return x
    
    def add_to_scrRgrMem(self, gameLog):
        """Add expereience to score regressor memory
        sb : ScoreBoard
        reward int
        """
        sbs = gameLog['scoreBoard'].tolist()
        for ii in range(len(sbs)-1):
            sb1 = sbs[ii]
            sb2 = sbs[ii+1]
            reward = sb2.getSum() - sb1.getSum()
            self.srm += [(sb1, reward, sb2)]
        self.truncate_srm()
    
    def add_to_rrRgrMem(self, gameLog):
        """Add expereience to score regressor memory
        sbs : ScoreBoards
        rrs : [dice, reroll, dice, reroll, dice]
        mem : list of (sb, att, dice0, reroll, dice1)
        reward int
        """
#        mem = (sb, att, dice0, reroll, dice1)
        for ii in range(13):
            sb = gameLog.loc[ii, 'scoreBoard']
            dice0, deci0 = gameLog.loc[ii, ['dice0', 'deci0']]
            dice1, deci1 = gameLog.loc[ii, ['dice1', 'deci1']]
            dice2 = gameLog.loc[ii, 'dice2']
#            sb = sbs[ii]
#            dice0, reroll0, dice1, reroll1, dice2 = rrs[ii]
            self.rrm += [(sb, 0, dice0, deci0, dice1)]
            self.rrm += [(sb, 1, dice1, deci0, dice2)]
        self.truncate_rrm()

    
    def truncate_srm(self):
        """Reduce srm length to lenScrReplayMem to the most recent memories.
        """
        self.srm = self.srm[-self.lenScrReplayMem:]
    
    def truncate_rrm(self):
        """Reduce srm length to lenScrReplayMem to the most recent memories.
        """
        self.rrm = self.rrm[-self.lenRrReplayMem:]
    
    def train(self, nGames,
              pOptCat=.3, pRandCat=0.1, pRatCat=10,
              pOptRr=.3, pRandRr=0.1, pRatRr=10):
        """Training the Player with nGames and based on the trainers moves.
    
        Extended description of function.
    
        Parameters
        ----------
        nGames : int
            Nomber of games
        trainerEnsemble : PlayerEnsemble
            Integer represents the weight of the specici players moves
            player is someone doing decisions; None is self
        pRandCat : 0 <= float <= 1
            Probability for a random move (choose cat).
            Equivalent to epsilon of eplison-greedy startegy.
            Increase this factor in case of anti-learning (i.e. systematically
            decreaings benchmarks to a bad decisions on purpose startegy)
        pRat : float
            chose an option for training which promisses a
            high score. A weighted choose is performed where
            the best option is pRat times as likely as
            the worst option.
    
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
        
        assert 0 <= pOptRr + pRandRr <= 1
        for gg in range(nGames):
            np.random.seed(self.nGames)
            if self.nGames == 0:
                tmpPlayer = PlayerRandom()
            else:
                def fct_choose_reroll(scoreBoard, dice, attempt):
                    sb = scoreBoard
                    rndNo = np.random.rand()
                    info = ''
                    if rndNo < pRandRr:
                        reroll = np.random.choice([True, False], size=5)
                        info = 'random'
                    elif pRatRr is None or rndNo < pRandRr+pOptRr:
                        reroll, info = self.choose_reroll(sb, dice, attempt)
                        info = 'optimal: ' + info
                    else:
                        opts = self.eval_options_reroll(sb, dice, attempt)
                        cs = [opt[0] for opt in opts]
                        ws = [opt[1] for opt in opts]
                        ws = np.array(ws) - np.amin(ws)
                        if np.amax(ws) > 0:
                            alpha = np.log(pRatRr)/np.amax(ws)
                            ws = np.exp(alpha*ws)
                            assert np.amin(ws) == 1
                            reroll = weighted_choice(cs, ws)
                            info = 'weighted: ' + str(opts)
                        else:
                            # all options are weighted equal
                            reroll = np.random.choice([True, False], size=5)
                            info = 'weighted->random: ' + str(opts)
                    return reroll, info
                
                def fct_choose_cat(scoreBoard, dice):
                    sb = scoreBoard
                    rndNo = np.random.rand()
                    info = ''
                    if rndNo < pRandCat:
                        info = 'random'
                        cat = np.random.choice(sb.open_cats())
                    elif pRatCat is None or rndNo < pRandCat+pOptCat:
                        
                        cat, info = self.choose_cat(sb, dice)
                        info = 'optimal: ' + info
                    else:
                        opts = self.eval_options_cat(sb, dice)
                        cs = [opt[0] for opt in opts]
                        ws = [opt[1] for opt in opts]
                        ws = np.array(ws) - np.amin(ws)
                        if np.amax(ws) > 0:
                            info = 'weighted: ' + str(opts)
                            alpha = np.log(pRatCat)/np.amax(ws)
                            ws = np.exp(alpha*ws)
                            cat = weighted_choice(cs, ws)
                        else:
                            info = 'weighted->random: ' + str(opts)
                            # all options are weighted equal
                            cat = np.random.choice(sb.open_cats())
                    return cat, info

                tmpPlayer = PlayerTemporary(
                        fct_choose_cat=fct_choose_cat,
                        fct_choose_reroll=fct_choose_reroll)

            game = Game(tmpPlayer)

            self.add_to_scrRgrMem(game.log)
            self.add_to_rrRgrMem(game.log)

            # scrRgr
            if self.nGames ==0:
                n_samples = len(self.srm)
                X = np.empty(shape=(n_samples, self.nFeat_scrRgr))
                y = np.empty(shape=(n_samples,))
                for nn in range(n_samples):
                    sb1, reward, sb2 = self.srm[nn]
                    X[nn, :] = self.encode_scrRgr_x(sb1).reshape(1,-1)
                    y[nn] = reward
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    self.scrRgr = self.scrRgr.fit(X, y)
                
            else:
                n_samples = self.lenScrMiniBatch
                X = np.empty(shape=(n_samples, self.nFeat_scrRgr))
                y = np.empty(shape=(n_samples,))
                for nn in range(n_samples):
                    ind = np.random.choice(list(range(len(self.srm))))
                    sb1, dirRew, sb2 = self.srm[ind]
                    X[nn, :] = self.encode_scrRgr_x(sb1).reshape(1,-1)
                    xSb2 = self.encode_scrRgr_x(sb2).reshape(1,-1)
                    futRew = self.scrRgr.predict(xSb2)[0]
                    y[nn] = dirRew + self.gamma * futRew
                for ii in range(self.nIterPartFit):
                    # perform multiple fit iterations
                    self.scrRgr = self.scrRgr.partial_fit(X, y)

            
            # rrRgr
            if self.nGames ==0:
                n_samples = len(self.rrm)
                X = np.empty(shape=(n_samples, self.nFeat_rrRgr))
                y = np.empty(shape=(n_samples,))
                for nn in range(n_samples):
                    sb, att, dice0, reroll, dice1 = self.rrm[nn]
                    X[nn, :] = (
                            self.encode_rrRgr_x(sb, att, dice0, reroll)
                            .reshape(1,-1))
                    y[nn] = 0
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    self.rrRgr = self.rrRgr.fit(X, y)
            else:
                n_samples = self.lenScrMiniBatch
                X = np.empty(shape=(n_samples, self.nFeat_rrRgr))
                y = np.empty(shape=(n_samples,))
                for nn in range(n_samples):
                    ind = np.random.choice(list(range(len(self.rrm))))
                    sb, att, diceOld, reroll, diceNew = self.rrm[ind]
                    X[nn, :] = (
                            self.encode_rrRgr_x(sb, att, diceOld, reroll)
                            .reshape(1,-1))
                    if att == 0:
#                        xNext = self.encode_rrRgr_x(sb, att, dice0, reroll)
#                            .reshape(1,-1))
                        opts = self.eval_options_reroll(sb, diceNew, 1)
                        y[nn] = opts[0][1]
                    elif att == 1:
                        opts = self.eval_options_cat(sb, diceNew)
                        y[nn] = opts[0][1]
                    else:
                        assert False
                self.rrRgr = self.rrRgr.partial_fit(X, y)
            self.nGames += 1
    
    def save(self, filename):
        """Store this instance as pickle"""
        pickle.dump(self, open(filename, "wb" ) )
    
    # https://stackoverflow.com/a/37658673
    def load(self, filename):
#        newObj = func(self)
#        self.__dict__.update(newObj.__dict__)
        player = pickle.load(open(filename, "rb"))
        self.__dict__.update(player.__dict__)


class PlayerAI_full_v1(AbstractPlayer):
    """AI Player: 3 regressor concept
    
    Regressors
    ----------
    rgrSC: forcasts the final score based on open categories and upper sum
    rgrRr: forcasts the DIFFERENCE to rgrScrCat based on dice to keep
    rgrEx: forcasts the expected score in each category based on a
                set of 0-5 dice, which are kept.
                This is used as auxiliary encoder information for rgrKeepDice.
    
    Categorie decision is made based on the direct reward 
    + the reward forcast of the resulting scoreboard.
    """
    name = 'PlayerAI_full_v1'
    def __init__(
            self,
#            rgrSCArgs={'hidden_layer_sizes':(20, 10)},
            rgrSCArgs={'hidden_layer_sizes':(40, 40), 'max_iter': 1000},
            rgrRrArgs={'hidden_layer_sizes':(20, 20), 'max_iter': 1000},
#            rgrRrArgs={'hidden_layer_sizes':(40, 40)},
            rgrExArgs={'activation': 'tanh',
                       'solver': 'adam',
                       'hidden_layer_sizes':(30, 40, 40, 30),
                       'max_iter': 1000},
#            nGamesPreplayMem=10,
#            nGamesPartFit=10,
            nGamesPreplayMem=200,
            nGamesPartFit=50,
            gamma=1,
            fn=None):
        """
        mlpRgrArgs : dict
            Arguments passed to MLPRegressor
        lenScrReplayMem : int
            Length of the replay memory for self.scrRgr training
        lenMiniBatch : int
            Number of samples used for each training iteration
        gamma : 0 <= gamma <= 1
            Damping factor for future rewards
        """
        super().__init__()
        
        self.rgrSC = MLPRegressor(**rgrSCArgs)
        self.rgrRr = MLPRegressor(**rgrRrArgs)
        self.rgrEx = MLPRegressor(**rgrExArgs)
        self.repMemSC = []
        self.repMemRr = []
        self.repMemEx = []
        self.nGamesPreplayMem = nGamesPreplayMem
        self.nGamesPartFit = nGamesPartFit
#        self.nRepPartFit = nRepPartFit
        self.gamma = gamma
        self.nGames = 0
        
        if fn is not None:
            self.load(fn)
            
        
        
    def choose_reroll(self, scoreBoard, dice, attempt):
        opts = self.eval_options_reroll(scoreBoard, dice, attempt)
        info = ''
        for opt in opts:
            info += '\t {:15} {:.2f}\n'.format(str(dice.keep(opt[0])), opt[1])
        return opts[0][0], info
#        return opts[0][0], str(opts)
    
    def choose_cat(self, scoreBoard, dice):
        opts = self.eval_options_cat(scoreBoard, dice)
        info = ''
        for opt in opts:
            info += '\t {:15} {:3.2f} {:3.2f} {:3.2f}\n'.format(
                    ScoreBoard.cats[opt[0]], *opt[1:])
        return opts[0][0], info
    
    def eval_options_cat(self, scoreBoard, dice):
        """Return a sorted list with the options to choose for cat and
        the expected restScore.
        returns: [(cat, score), (cat, score), ...]
        """
        opts = []

        for cat in scoreBoard.open_cats():
            score, bonus = scoreBoard.check_points(dice, cat)
#            assert bonus == 0, str(scoreBoard) + str(dice)
            directReward = score + bonus
            
            # additionally consider the resulting state of the score board
            tmpSB = scoreBoard.copy()
            tmpSB.add(dice, cat)
            futureReward = self.predict_SC(tmpSB)
            
            reward = directReward + self.gamma * futureReward
            opts += [(cat, reward, directReward, futureReward)]
            
            
        opts = sorted(opts, key=lambda x: x[1], reverse=True)
        return opts
    
    
    
    def eval_options_reroll(self, sb, dice, att):
        opts = []
        keepDices = []
        for reroll in product([True, False], repeat=5):
            # avoid unnecessary rerolls ([1, 2r, 2, 3, 4] and [1, 2, 2r, 3, 4])
            keepDice = dice.keep(reroll).vals
            if arreq_in_list(keepDice, keepDices):
                continue
            else:
                keepDices += [keepDice]

            reward = self.predict_Rr(sb, att, dice, reroll)
            opts += [(reroll, reward)]

        opts = sorted(opts, key=lambda x: x[1], reverse=True)
        return opts
    
    @staticmethod
    def encode_bonus(scoreBoard):
#        upSum = scoreBoard.getUpperSum()
#        if upSum >= 63 or np.sum(scoreBoard.mask[:6])==0:
##            x[13] = (5*(2+3+4+5+6) - 63) / 5  # max posible values
#            x[13] = -63/5  # min possible value
#        else:
#            # nbp: needed bonus progress (i.e. 3 in each cat)
#            nbp = np.sum(
#                    np.logical_not(scoreBoard.mask[:6]).astype(int)
#                    * np.array(range(1,7)) * 3)
#            mrbp = (63 - nbp) / 3 * 5  # max reachable bonus points
#            x[13] = (upSum - nbp) / mrbp
##            lp(scoreBoard.scores[:6])
##            lp(nbp, scoreBoard.getUpperSum(), x[20])
#        # prevent impossible bonus attempts
#        if x[13] < -1:
#            x[13] = -63/5
        return scoreBoard.getUpperSum() / 63
    
    @property
    def nFeat_SC(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 1
    def encode_SC(self, scoreBoard):
        """Encodes a scoreboard to a numpy array,
        which is used as the input layer
        """
        x = np.zeros(shape=(self.nFeat_SC))
        x[:13] = scoreBoard.mask.astype(int)
        x[13] = self.encode_bonus(scoreBoard)
        return x
    def predict_SC(self, scoreBoard):
        """Includes exception handling for direct predict call"""
        x = self.encode_SC(scoreBoard)
        try:
            y = self.rgrSC.predict(x.reshape(1, -1))[0]
        except NotFittedError:
            y = 0
        return y
    def repMemSC_xy(self, ind):
        """Constructs input (x), output (y) tuple from replay memory."""
        sbOld, rew, sbNew = self.repMemSC[ind]
        x = self.encode_SC(sbOld)
        y = rew + self.gamma * self.predict_SC(sbNew)
        return x, y
    
    @property
    def nFeat_Rr(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 13 + 1 + 1
    def encode_Rr(self, scoreBoard, attempt, dice, deciRr):
        """Encodes a scoreboard to a numpy array,
        which is used as the input layer
        13: open cats
        13: exp score in each cat
        1: attempt
        1: bonus
        """
        x = np.zeros(shape=(self.nFeat_Rr))
        x[:13] = scoreBoard.mask.astype(int)
        x[13:26] = self.predict_Ex(dice, deciRr)
        x[13:26] *= x[:13]
#        x[13:26], _ = ScoreBoard.stat_cat_score(dice.keep(deciRr))
        x[26] = attempt
        x[27] = self.encode_bonus(scoreBoard)
        return x
    def predict_Rr(self, scoreBoard, attempt, dice, deciRr):
        """Includes exception handling for direct predict call"""
        x = self.encode_Rr(scoreBoard, attempt, dice, deciRr)
        try:
            y = self.rgrRr.predict(x.reshape(1, -1))[0]
        except NotFittedError:
            y = 0
        return y
    def repMemRr_xy(self, ind):
        """Constructs input (x), output (y) tuple from replay memory."""
        sb, attempt, diceOld, deciRr, diceNew = self.repMemRr[ind]
        x = self.encode_Rr(sb, attempt, diceOld, deciRr)
        if attempt == 0:
            opts = self.eval_options_reroll(sb, diceNew, 1)
            y = self.gamma * opts[0][1]
        else:
            assert attempt == 1
            opts = self.eval_options_cat(sb, diceNew)
            # evaluate SC forfast before reroll
            yOld = self.predict_SC(sb)
            y = self.gamma * (opts[0][1] - yOld)
        return x, y
    
    @property
    def nFeat_Ex(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 5 + 1 + 6 + 1 + 5 + 2 + 4
    def encode_Ex(self, dice, deciRr):
        """Encodes a scoreboard to a numpy array,
        which is used as the input layer
        
        input layer:
            5: dice (values: 1-6 / to be rerolled: 0)
            1: number of kept dice
            6: histogramm (cats 1-6) * score
            1: sum(dice)
            5: number of 1ofAKind, double, triple, 4ofAKind, 5ofAKind
            2: small straight indicators
            4: large straight indicators
        """
        keepDice = dice.vals[np.logical_not(deciRr)]
        x = np.zeros(shape=(self.nFeat_Ex))
        x[:len(keepDice)] = keepDice
        x[5] = len(keepDice)
        hist = np.histogram(keepDice, bins=np.linspace(.5,6.5,7))[0]
        x[6:12] = hist * np.array([1,2,3,4,5,6])
        x[12] = np.sum(keepDice)
        for ii in range(1, 6):  # number of single, double, ...
            x[12 + ii] = len(hist[hist == ii])  # x[13]-x[17]
        # small straight
        if set([1, 2, 3]) <= set(keepDice) or set([4, 5, 6]) <= set(keepDice):
            x[18] = 1  # 3 in a row, one open end
        if set([2, 3, 4]) <= set(keepDice) or set([3, 4, 5]) <= set(keepDice):
            x[19] = 1  # 3 in a row, two open ends
        # large straight
        for nr in range(2):
            numbsInRange = set(keepDice) & set(range(1+nr, 6+nr))
            x[20 + 2*nr] = len(numbsInRange)  # number of different dice
            segs = split_in_consec_ints(numbsInRange)
            segs = [len(sl) for sl in segs]
            x[21 + 2*nr] = np.amax(segs)  # len of max consequtive sequence
                
        # how many numbers occupied in range 1-5
        # max number of conseccutive in range 1-5
        return x
    def predict_Ex(self, dice, deciRr):
        """Includes exception handling for direct predict call"""
        x = self.encode_Ex(dice, deciRr)
        try:
            y = self.rgrEx.predict(x.reshape(1, -1))[0]
        except NotFittedError:
            y = np.zeros(shape=(13))
        return y
    def repMemEx_xy(self, ind):
        """Constructs input (x), output (y) tuple from replay memory."""
        diceOld, deci, diceNew = self.repMemEx[ind]
        return self.encode_Ex_xy(diceOld, deci, diceNew)
    def encode_Ex_y(self, dice):
        """Evaluate the score in each cat based on dice
        dice : can be a 0 to 5 dice Dice object"""
        y = np.zeros(shape=(13))
        sb = ScoreBoard()
        for cc in range(13):
            y[cc], bonus = sb.check_points(dice, cc)
#        lp(dice, y)
        return y
    def encode_Ex_xy(self, diceOld, deciRr, diceNew):
        x = self.encode_Ex(diceOld, deciRr)
        y = self.encode_Ex_y(diceNew)
#        y = np.zeros(shape=(13))
#        sb = ScoreBoard()
#        for cc in range(13):
#            y[cc], bonus = sb.check_points(diceNew, cc)
        return x, y
    
    def aux_Ex_genTrainingTuple(self, seed=None):
        """generates diceOld, deci, diceNew pairs, 
        which can be used for training or testing
        
        n : int
            number of pairs
        seed : int
            random numbers seed
        
        returns:
            list of (diceOld, deci, diceNew) tuples
        """
        if seed is not None:
            np.random.seed(seed)
#        lst = []
#        for nn in range(n):
#            diceOld = Dice()
#            deci = np.random.choice([True, False], size=5)
#            diceNew = diceOld.roll(deci)
#            lst += [(diceOld, deci, diceNew)]
#        return lst
        diceOld = Dice()
        deci = np.random.choice([True, False], size=5)
        diceNew = diceOld.roll(deci)
        return diceOld, deci, diceNew
    def aux_Ex_train(self, n, facMC=2, seed=None, optRgrParas=False):
        """trains rgrEx separately
        
        n : int
            number of pairs
        facMC : float
            number of Monte Carlo trials / number of possible combinations
        seed : int
            random numbers seed
        optRgrParas : bool
            To test different Regressor Layouts.
            These can be used for self.rgrEx then.
        """
        if seed is not None:
            np.random.seed(seed)
#        lst = self.aux_Ex_genPairs(n, seed)
        X = np.empty(shape=(n, self.nFeat_Ex))
        y = np.empty(shape=(n, 13))
        
        for nn in range(n):
            diceOld, deci, diceNew = self.aux_Ex_genTrainingTuple()
            X[nn, :], y[nn, :] = self.encode_Ex_xy(diceOld, deci, diceNew)
        self.rgrEx = self.rgrEx.fit(X, y)
        
#        for nn in range(n):
#            diceOld = Dice()
#            deciRr = np.random.choice([True, False], size=5)
#            X[nn, :] = self.encode_Ex(diceOld, deciRr)
#            
#            # Monte Carlo simulation to get expection values in each cat
#            nMC = int(np.rint(facMC * 6**np.sum(deciRr)))
#            if np.sum(deciRr) == 0:
#                nMC = 1
#            yMC = np.empty(shape=(nMC, 13))
#            for mm in range(nMC):
#                diceNew = diceOld.roll(deciRr)
#                yMC[mm, :] = self.encode_Ex_y(diceNew)
#            y[nn, :] = np.mean(yMC, axis=0)
#            
##            diceOld, deci, diceNew = self.aux_Ex_genTrainingTuple()
##            X[nn, :], y[nn, :] = self.encode_Ex_xy(diceOld, deci, diceNew)
##            assert np.isfinite(X[nn, :]).all(), str(X[nn, :])
##            assert np.isfinite(y[nn, :]).all(), str(y[nn, :])
#        self.rgrEx = self.rgrEx.fit(X, y)
        
        if optRgrParas:
            # https://stackoverflow.com/a/46031556
            from sklearn.model_selection import GridSearchCV
            param_grid = [
                    {'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                     'solver' : ['lbfgs', 'sgd', 'adam'],
                     'hidden_layer_sizes': [
                             (10,), (20,), (30,),
                             (10, 10), (20, 20), (30, 30),
#                             (10, 10, 10), (20, 20, 20), (30, 30, 30),
#                             (40, 40, 40),
#                             (20, 40, 40, 20, 30), (20, 30, 40, 40, 20),
#                             (40, 30, 20, 40, 20), (20, 40, 30, 40, 20),
                             ]
                     }
                    ]
            rgr = GridSearchCV(MLPRegressor(), param_grid, cv=5)#, scoring='accuracy')
            assert np.isfinite(X).all(), str(X)
            assert np.isfinite(y).all(), str(y)
            rgr.fit(X,y)
            lp("Best parameters set found on development set:")
            lp(rgr.best_params_)
        
    def aux_Ex_benchmark(self, n, seed=None):
        """trains rgrEx separately
        
        n : int
            number of pairs
        nMC : int
            number of Monte Carlo trials to verify
        facMC : float
            number of Monte Carlo trials / number of possible combinations
        seed : int
            random numbers seed
        """
        bmAbs = []
        bmRel = []
        for nn in range(n):
            diceOld = Dice()
            deciRr = np.random.choice([True, False], size=5)
            x = self.encode_Ex(diceOld, deciRr)
            y = self.rgrEx.predict(x.reshape(1, -1))[0]
            
            diceKeep = diceOld.keep(deciRr)
            meanComb, semComb = ScoreBoard.stat_cat_score(diceKeep)
            
#            nRr = int(np.sum(deciRr))
#            nCombs = 6**nRr
#
#            yCombs = np.empty(shape=(nCombs, 13))
#            for mm, comb in enumerate(product([1, 2, 3, 4, 5, 6], repeat=nRr)):
#                diceNew = np.copy(diceOld.vals)
#                diceNew[deciRr] = comb
#                diceNew = Dice(diceNew)
##                lp(diceOld, deciRr, comb)
#                yCombs[mm, :] = self.encode_Ex_y(diceNew)
#            meanComb = np.mean(yCombs, axis=0)
#            semComb = np.std(yCombs, axis=0) / nCombs**.5
            dist = y - meanComb
            dist = np.where(abs(dist) <= semComb, 0, dist)
            dist = np.where(dist > 0, dist-semComb, dist)
            dist = np.where(dist < 0, -dist+semComb, dist)
            
            assert (dist >= 0).all()
#            lp(diceKeep)
#            lp(y)
#            lp(meanComb)
#            lp(semComb)
#            lp(dist)
            
            bmAbs += [dist]
            norm = meanComb #(y + meanComb) / 2
            norm = np.where(norm < 1, 1, norm)
            bmRel += [dist/norm]

        return np.mean(bmAbs, axis=0), np.mean(bmRel, axis=0)
#        return np.mean(bmAbs, axis=0), np.std(benchmark, axis=0)
    
    def aux_Ex_benchmark_bak(self, n, nMC, seed=None):
        """trains rgrEx separately
        
        n : int
            number of pairs
        nMC : int
            number of Monte Carlo trials to verify
        seed : int
            random numbers seed
        """
        benchmark = []
        for nn in range(n):
            diceOld = Dice()
            deciRr = np.random.choice([True, False], size=5)
            x = self.encode_Ex(diceOld, deciRr)
            y = self.rgrEx.predict(x.reshape(1, -1))[0]
            yMC = np.empty(shape=(nMC, 13))
            for mm in range(nMC):
                diceNew = diceOld.roll(deciRr)
                _, yMC[mm, :] = self.encode_Ex_xy(diceOld, deciRr, diceNew)
            yMC = np.mean(yMC, axis=0)
            benchmark += [np.linalg.norm(y-yMC)]
        return np.mean(benchmark), np.std(benchmark)
    
    def to_repMem(self, gameLog):
        """Add expereience to score regressor replay memory
        
        if possible in format:
        state1, action, reward, state2
        
        formats
        -------
        rgrScrCat: (sb, rew, sb)
        rgrDeltaRr: (sb, attempt, dice, deciRr, dice)
        rgrChances: (dice, deciRr, dice)
        
        gameLog : Game.log
        """
        for ii in range(13):
            sb1 = gameLog.loc[ii, 'scoreBoard']
            sb2 = gameLog.loc[ii+1, 'scoreBoard']
            reward = sb2.getSum() - sb1.getSum()
            self.repMemSC += [(sb1, reward, sb2)]
            
            dice0, deci0 = gameLog.loc[ii, ['dice0', 'deci0']]
            dice1, deci1 = gameLog.loc[ii, ['dice1', 'deci1']]
            dice2 = gameLog.loc[ii, 'dice2']
            self.repMemRr += [(sb1, 0, dice0, deci0, dice1)]
            self.repMemRr += [(sb1, 1, dice1, deci1, dice2)]
            
            self.repMemEx += [(dice0, deci0, dice1), (dice1, deci1, dice2)]
            
        self.repMemSC = self.repMemSC[-self.nGamesPreplayMem*13:]
        self.repMemRr = self.repMemRr[-self.nGamesPreplayMem*26:]
        self.repMemEx = self.repMemEx[-self.nGamesPreplayMem*26:]
    

    
    def train(self, nGames,
              expl_cat_fct=explore_boltzmann,
              expl_cat_params={'minMaxRat': 5000},
              expl_rr_fct=explore_boltzmann,
              expl_rr_params={'minMaxRat': 50000},
#              pOptCat=.3, pRandCat=0.1, pRatCat=10,
#              pOptRr=.9, pRandRr=0.05, pRatRr=10
              ):
        """Training the Player with nGames and based on the trainers moves.
    
        Extended description of function.
    
        Parameters
        ----------
        nGames : int
            Nomber of games
        trainerEnsemble : PlayerEnsemble
            Integer represents the weight of the specici players moves
            player is someone doing decisions; None is self
        pRandCat : 0 <= float <= 1
            Probability for a random move (choose cat).
            Equivalent to epsilon of eplison-greedy startegy.
            Increase this factor in case of anti-learning (i.e. systematically
            decreaings benchmarks to a bad decisions on purpose startegy)
        pRat : float
            chose an option for training which promisses a
            high score. A weighted choose is performed where
            the best option is pRat times as likely as
            the worst option.
    
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
        if self.nGames == 0:
            # initiallize rgrEx with some random samples first
            np.random.seed(self.nGames)
            self.aux_Ex_train(n=100000)
        
#        assert 0 <= pOptRr + pRandRr <= 1
        for gg in range(nGames):
            np.random.seed(self.nGames)
            if self.nGames == 0:
                tmpPlayer = PlayerRandom()
            else:
                def fct_choose_reroll(scoreBoard, dice, attempt):
                    sb = scoreBoard
                    info = ''
                    
                    opts = self.eval_options_reroll(sb, dice, attempt)
                    actions = [opt[0] for opt in opts]
                    qs = [opt[1] for opt in opts]
                    reroll = expl_rr_fct(actions, qs, **expl_rr_params)
                    
                    return reroll, info
                
                def fct_choose_cat(scoreBoard, dice):
                    sb = scoreBoard
                    info = ''
                    
                    opts = self.eval_options_cat(sb, dice)
#                    opts = np.array(opts)
                    actions = [opt[0] for opt in opts]
                    qs = [opt[1] for opt in opts]
                    cat = expl_cat_fct(actions, qs, **expl_cat_params)
                    
                    return cat, info

                tmpPlayer = PlayerTemporary(
                        fct_choose_cat=fct_choose_cat,
                        fct_choose_reroll=fct_choose_reroll)

            game = Game(tmpPlayer)
            self.to_repMem(game.log)
            
            #rgrSC
            n_samples = self.nGamesPartFit * 13
            X = np.empty(shape=(n_samples, self.nFeat_SC))
            y = np.empty(shape=(n_samples,))
            allInds = list(range(len(self.repMemSC)))
            replace=False
            if len(allInds) < n_samples:
                replace = True
            inds = np.random.choice(allInds, size=n_samples, replace=replace)
            for nn, ind in enumerate(inds):
                X[nn, :], y[nn] = self.repMemSC_xy(ind)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
#                self.rgrSC = self.rgrSC.partial_fit(X, y)
                self.rgrSC.partial_fit(X, y)
            
            #rgrEx
            n_samples = self.nGamesPartFit * 26
            X = np.empty(shape=(n_samples, self.nFeat_Ex))
            y = np.empty(shape=(n_samples, 13))
            allInds = list(range(len(self.repMemEx)))
            replace=False
            if len(allInds) < n_samples:
                replace = True
            inds = np.random.choice(allInds, size=n_samples, replace=replace)
            for nn, ind in enumerate(inds):
                X[nn, :], y[nn, :] = self.repMemEx_xy(ind)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
#                self.rgrEx = self.rgrEx.partial_fit(X, y)
                self.rgrEx.partial_fit(X, y)
            
            #rgrRr
            n_samples = self.nGamesPartFit * 26
            X = np.empty(shape=(n_samples, self.nFeat_Rr))
            y = np.empty(shape=(n_samples,))
            allInds = list(range(len(self.repMemRr)))
            replace=False
            if len(allInds) < n_samples:
                replace = True
            inds = np.random.choice(allInds, size=n_samples, replace=replace)
            for nn, ind in enumerate(inds):
                X[nn, :], y[nn] = self.repMemRr_xy(ind)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
#                self.rgrRr = self.rgrRr.partial_fit(X, y)
                self.rgrRr.partial_fit(X, y)

            self.nGames += 1
    
    def save(self, filename):
        """Store this instance as pickle"""
        # delete loss curves to save filesize
        self.rgrSC.loss_curve_ = []
        self.rgrRr.loss_curve_ = []
        self.rgrEx.loss_curve_ = []
        
        pickle.dump(self, open(filename, "wb" ) )
    
    # https://stackoverflow.com/a/37658673
    def load(self, filename):
        try:
            player = pickle.load(open(filename, "rb"))
        except FileNotFoundError:
#            print('Coluld not loaded player! File:', filename, 'not found!')
            return
        else:
            self.__dict__.update(player.__dict__)
#            print('Loaded player from file:', filename)
            



class PlayerAI_full_v2(AbstractPlayer):
    """AI Player: 3 regressor concept
    
    Regressors
    ----------
    rgrSC: forcasts the final score based on open categories and upper sum
    rgrRr: forcasts the DIFFERENCE to rgrScrCat based on dice to keep
    rgrEx: forcasts the expected score in each category based on a
                set of 0-5 dice, which are kept.
                This is used as auxiliary encoder information for rgrKeepDice.
    
    Categorie decision is made based on the direct reward 
    + the reward forcast of the resulting scoreboard.
    """
    name = 'PlayerAI_full_v2'
    def __init__(
            self,
#            rgrSCArgs={'hidden_layer_sizes':(20, 10)},
            rgrSCArgs={'hidden_layer_sizes':(40, 40), 'max_iter': 1000},
            rgrRrArgs={'hidden_layer_sizes':(20, 20), 'max_iter': 1000},
            rgrExArgs={'activation': 'tanh',
                       'solver': 'adam',
                       'hidden_layer_sizes':(30, 40, 40, 30),
                       'max_iter': 1000},
#            nGamesPreplayMem=10,
#            nGamesPartFit=10,
            nGamesPreplayMem=200,
            nGamesPartFit=50,
            gamma=1,
            fn=None):
        """
        mlpRgrArgs : dict
            Arguments passed to MLPRegressor
        lenScrReplayMem : int
            Length of the replay memory for self.scrRgr training
        lenMiniBatch : int
            Number of samples used for each training iteration
        gamma : 0 <= gamma <= 1
            Damping factor for future rewards
        """
        super().__init__()
        
        self.rgrSC = MLPRegressor(**rgrSCArgs)
        self.rgrRr = MLPRegressor(**rgrRrArgs)
        self.rgrEx = MLPRegressor(**rgrExArgs)
        self.repMemSC = []
        self.repMemRr = []
        self.repMemEx = []
        self.nGamesPreplayMem = nGamesPreplayMem
        self.nGamesPartFit = nGamesPartFit
#        self.nRepPartFit = nRepPartFit
        self.gamma = gamma
        self.nGames = 0
        
        self.rgrBaks = None
        
        self.scsp = statCatScorePredictor_exact()
        
        if fn is not None:
            self.load(fn)
            
        
        
    def choose_reroll(self, scoreBoard, dice, attempt):
        opts = self.eval_options_reroll(scoreBoard, dice, attempt)
        info = ''
        for opt in opts:
            info += '\t {:15} {:.2f}\n'.format(str(dice.keep(opt[0])), opt[1])
        return opts[0][0], info
#        return opts[0][0], str(opts)
    
    def choose_cat(self, scoreBoard, dice):
        opts = self.eval_options_cat(scoreBoard, dice)
        info = ''
        for opt in opts:
            info += '\t {:15} {:3.2f} {:3.2f} {:3.2f}\n'.format(
                    ScoreBoard.cats[opt[0]], *opt[1:])
        return opts[0][0], info
    
    def eval_options_cat(self, scoreBoard, dice):
        """Return a sorted list with the options to choose for cat and
        the expected restScore.
        returns: [(cat, score), (cat, score), ...]
        """
        opts = []

        for cat in scoreBoard.open_cats():
            score, bonus = scoreBoard.check_points(dice, cat)
#            assert bonus == 0, str(scoreBoard) + str(dice)
            directReward = score + bonus
            
            # additionally consider the resulting state of the score board
            tmpSB = scoreBoard.copy()
            tmpSB.add(dice, cat)
            futureReward = self.predict_SC(tmpSB)
            
            reward = directReward + self.gamma * futureReward
            opts += [(cat, reward, directReward, futureReward)]
            
            
        opts = sorted(opts, key=lambda x: x[1], reverse=True)
        return opts
    
    
    
    def eval_options_reroll(self, sb, dice, att):
        opts = []
        keepDices = []
        for reroll in product([True, False], repeat=5):
            # avoid unnecessary rerolls ([1, 2r, 2, 3, 4] and [1, 2, 2r, 3, 4])
            keepDice = dice.keep(reroll).vals
            if arreq_in_list(keepDice, keepDices):
                continue
            else:
                keepDices += [keepDice]

            reward = self.predict_Rr(sb, att, dice, reroll)
            opts += [(reroll, reward)]

        opts = sorted(opts, key=lambda x: x[1], reverse=True)
        return opts
    
    @staticmethod
    def encode_bonus(scoreBoard):
#        upSum = scoreBoard.getUpperSum()
#        if upSum >= 63 or np.sum(scoreBoard.mask[:6])==0:
##            x[13] = (5*(2+3+4+5+6) - 63) / 5  # max posible values
#            x[13] = -63/5  # min possible value
#        else:
#            # nbp: needed bonus progress (i.e. 3 in each cat)
#            nbp = np.sum(
#                    np.logical_not(scoreBoard.mask[:6]).astype(int)
#                    * np.array(range(1,7)) * 3)
#            mrbp = (63 - nbp) / 3 * 5  # max reachable bonus points
#            x[13] = (upSum - nbp) / mrbp
##            lp(scoreBoard.scores[:6])
##            lp(nbp, scoreBoard.getUpperSum(), x[20])
#        # prevent impossible bonus attempts
#        if x[13] < -1:
#            x[13] = -63/5
        return scoreBoard.getUpperSum() / 63
    
    @property
    def nFeat_SC(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 1
    def encode_SC(self, scoreBoard):
        """Encodes a scoreboard to a numpy array,
        which is used as the input layer
        """
        x = np.zeros(shape=(self.nFeat_SC))
        x[:13] = scoreBoard.mask.astype(int)
        x[13] = self.encode_bonus(scoreBoard)
        return x
    def predict_SC(self, scoreBoard):
        """Includes exception handling for direct predict call"""
        x = self.encode_SC(scoreBoard)
        try:
            y = self.rgrSC.predict(x.reshape(1, -1))[0]
        except NotFittedError:
            y = 0
        return y
    def repMemSC_xy(self, ind):
        """Constructs input (x), output (y) tuple from replay memory."""
        sbOld, rew, sbNew = self.repMemSC[ind]
        x = self.encode_SC(sbOld)
        y = rew + self.gamma * self.predict_SC(sbNew)
        return x, y
    
    @property
    def nFeat_Rr(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 13 + 1 + 1
    def encode_Rr(self, scoreBoard, attempt, dice, deciRr):
        """Encodes a scoreboard to a numpy array,
        which is used as the input layer
        13: open cats
        13: exp score in each cat
        1: attempt
        1: bonus
        """
        x = np.zeros(shape=(self.nFeat_Rr))
        x[:13] = scoreBoard.mask.astype(int)
#        x[13:26] = self.predict_Ex(dice, deciRr)
        x[13:26] = self.scsp.predict(dice.keep(deciRr))
        x[13:26] *= x[:13]
#        x[13:26], _ = ScoreBoard.stat_cat_score(dice.keep(deciRr))
        x[26] = attempt
        x[27] = self.encode_bonus(scoreBoard)
        return x
    def predict_Rr(self, scoreBoard, attempt, dice, deciRr):
        """Includes exception handling for direct predict call"""
        x = self.encode_Rr(scoreBoard, attempt, dice, deciRr)
        try:
            y = self.rgrRr.predict(x.reshape(1, -1))[0]
        except NotFittedError:
            y = 0
        return y
    def repMemRr_xy(self, ind):
        """Constructs input (x), output (y) tuple from replay memory."""
        sb, attempt, diceOld, deciRr, diceNew = self.repMemRr[ind]
        x = self.encode_Rr(sb, attempt, diceOld, deciRr)
        if attempt == 0:
            opts = self.eval_options_reroll(sb, diceNew, 1)
            y = self.gamma * opts[0][1]
        else:
            assert attempt == 1
            opts = self.eval_options_cat(sb, diceNew)
            # evaluate SC forfast before reroll
            yOld = self.predict_SC(sb)
            y = self.gamma * (opts[0][1] - yOld)
        return x, y
    
    @property
    def nFeat_Ex(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 5 + 1 + 6 + 1 + 5 + 2 + 4
    def encode_Ex(self, dice, deciRr):
        """Encodes a scoreboard to a numpy array,
        which is used as the input layer
        
        input layer:
            5: dice (values: 1-6 / to be rerolled: 0)
            1: number of kept dice
            6: histogramm (cats 1-6) * score
            1: sum(dice)
            5: number of 1ofAKind, double, triple, 4ofAKind, 5ofAKind
            2: small straight indicators
            4: large straight indicators
        """
        keepDice = dice.vals[np.logical_not(deciRr)]
        x = np.zeros(shape=(self.nFeat_Ex))
        x[:len(keepDice)] = keepDice
        x[5] = len(keepDice)
        hist = np.histogram(keepDice, bins=np.linspace(.5,6.5,7))[0]
        x[6:12] = hist * np.array([1,2,3,4,5,6])
        x[12] = np.sum(keepDice)
        for ii in range(1, 6):  # number of single, double, ...
            x[12 + ii] = len(hist[hist == ii])  # x[13]-x[17]
        # small straight
        if set([1, 2, 3]) <= set(keepDice) or set([4, 5, 6]) <= set(keepDice):
            x[18] = 1  # 3 in a row, one open end
        if set([2, 3, 4]) <= set(keepDice) or set([3, 4, 5]) <= set(keepDice):
            x[19] = 1  # 3 in a row, two open ends
        # large straight
        for nr in range(2):
            numbsInRange = set(keepDice) & set(range(1+nr, 6+nr))
            x[20 + 2*nr] = len(numbsInRange)  # number of different dice
            segs = split_in_consec_ints(numbsInRange)
            segs = [len(sl) for sl in segs]
            x[21 + 2*nr] = np.amax(segs)  # len of max consequtive sequence
                
        # how many numbers occupied in range 1-5
        # max number of conseccutive in range 1-5
        return x
    def predict_Ex(self, dice, deciRr):
        """Includes exception handling for direct predict call"""
        x = self.encode_Ex(dice, deciRr)
        try:
            y = self.rgrEx.predict(x.reshape(1, -1))[0]
        except NotFittedError:
            y = np.zeros(shape=(13))
        return y
    def repMemEx_xy(self, ind):
        """Constructs input (x), output (y) tuple from replay memory."""
        diceOld, deci, diceNew = self.repMemEx[ind]
        return self.encode_Ex_xy(diceOld, deci, diceNew)
    def encode_Ex_y(self, dice):
        """Evaluate the score in each cat based on dice
        dice : can be a 0 to 5 dice Dice object"""
        y = np.zeros(shape=(13))
        sb = ScoreBoard()
        for cc in range(13):
            y[cc], bonus = sb.check_points(dice, cc)
#        lp(dice, y)
        return y
    def encode_Ex_xy(self, diceOld, deciRr, diceNew):
        x = self.encode_Ex(diceOld, deciRr)
        y = self.encode_Ex_y(diceNew)
#        y = np.zeros(shape=(13))
#        sb = ScoreBoard()
#        for cc in range(13):
#            y[cc], bonus = sb.check_points(diceNew, cc)
        return x, y
    
    def aux_Ex_genTrainingTuple(self, seed=None):
        """generates diceOld, deci, diceNew pairs, 
        which can be used for training or testing
        
        n : int
            number of pairs
        seed : int
            random numbers seed
        
        returns:
            list of (diceOld, deci, diceNew) tuples
        """
        if seed is not None:
            np.random.seed(seed)
#        lst = []
#        for nn in range(n):
#            diceOld = Dice()
#            deci = np.random.choice([True, False], size=5)
#            diceNew = diceOld.roll(deci)
#            lst += [(diceOld, deci, diceNew)]
#        return lst
        diceOld = Dice()
        deci = np.random.choice([True, False], size=5)
        diceNew = diceOld.roll(deci)
        return diceOld, deci, diceNew
    def aux_Ex_train(self, n, facMC=2, seed=None, optRgrParas=False):
        """trains rgrEx separately
        
        n : int
            number of pairs
        facMC : float
            number of Monte Carlo trials / number of possible combinations
        seed : int
            random numbers seed
        optRgrParas : bool
            To test different Regressor Layouts.
            These can be used for self.rgrEx then.
        """
        if seed is not None:
            np.random.seed(seed)
#        lst = self.aux_Ex_genPairs(n, seed)
        X = np.empty(shape=(n, self.nFeat_Ex))
        y = np.empty(shape=(n, 13))
        
        for nn in range(n):
            diceOld, deci, diceNew = self.aux_Ex_genTrainingTuple()
            X[nn, :], y[nn, :] = self.encode_Ex_xy(diceOld, deci, diceNew)
        self.rgrEx = self.rgrEx.fit(X, y)
        
#        for nn in range(n):
#            diceOld = Dice()
#            deciRr = np.random.choice([True, False], size=5)
#            X[nn, :] = self.encode_Ex(diceOld, deciRr)
#            
#            # Monte Carlo simulation to get expection values in each cat
#            nMC = int(np.rint(facMC * 6**np.sum(deciRr)))
#            if np.sum(deciRr) == 0:
#                nMC = 1
#            yMC = np.empty(shape=(nMC, 13))
#            for mm in range(nMC):
#                diceNew = diceOld.roll(deciRr)
#                yMC[mm, :] = self.encode_Ex_y(diceNew)
#            y[nn, :] = np.mean(yMC, axis=0)
#            
##            diceOld, deci, diceNew = self.aux_Ex_genTrainingTuple()
##            X[nn, :], y[nn, :] = self.encode_Ex_xy(diceOld, deci, diceNew)
##            assert np.isfinite(X[nn, :]).all(), str(X[nn, :])
##            assert np.isfinite(y[nn, :]).all(), str(y[nn, :])
#        self.rgrEx = self.rgrEx.fit(X, y)
        
        if optRgrParas:
            # https://stackoverflow.com/a/46031556
            from sklearn.model_selection import GridSearchCV
            param_grid = [
                    {'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                     'solver' : ['lbfgs', 'sgd', 'adam'],
                     'hidden_layer_sizes': [
                             (10,), (20,), (30,),
                             (10, 10), (20, 20), (30, 30),
#                             (10, 10, 10), (20, 20, 20), (30, 30, 30),
#                             (40, 40, 40),
#                             (20, 40, 40, 20, 30), (20, 30, 40, 40, 20),
#                             (40, 30, 20, 40, 20), (20, 40, 30, 40, 20),
                             ]
                     }
                    ]
            rgr = GridSearchCV(MLPRegressor(), param_grid, cv=5)#, scoring='accuracy')
            assert np.isfinite(X).all(), str(X)
            assert np.isfinite(y).all(), str(y)
            rgr.fit(X,y)
            lp("Best parameters set found on development set:")
            lp(rgr.best_params_)
        
    def aux_Ex_benchmark(self, n, seed=None):
        """trains rgrEx separately
        
        n : int
            number of pairs
        nMC : int
            number of Monte Carlo trials to verify
        facMC : float
            number of Monte Carlo trials / number of possible combinations
        seed : int
            random numbers seed
        """
        bmAbs = []
        bmRel = []
        for nn in range(n):
            diceOld = Dice()
            deciRr = np.random.choice([True, False], size=5)
            x = self.encode_Ex(diceOld, deciRr)
            y = self.rgrEx.predict(x.reshape(1, -1))[0]
            
            diceKeep = diceOld.keep(deciRr)
            meanComb, semComb = ScoreBoard.stat_cat_score(diceKeep)
            
#            nRr = int(np.sum(deciRr))
#            nCombs = 6**nRr
#
#            yCombs = np.empty(shape=(nCombs, 13))
#            for mm, comb in enumerate(product([1, 2, 3, 4, 5, 6], repeat=nRr)):
#                diceNew = np.copy(diceOld.vals)
#                diceNew[deciRr] = comb
#                diceNew = Dice(diceNew)
##                lp(diceOld, deciRr, comb)
#                yCombs[mm, :] = self.encode_Ex_y(diceNew)
#            meanComb = np.mean(yCombs, axis=0)
#            semComb = np.std(yCombs, axis=0) / nCombs**.5
            dist = y - meanComb
            dist = np.where(abs(dist) <= semComb, 0, dist)
            dist = np.where(dist > 0, dist-semComb, dist)
            dist = np.where(dist < 0, -dist+semComb, dist)
            
            assert (dist >= 0).all()
#            lp(diceKeep)
#            lp(y)
#            lp(meanComb)
#            lp(semComb)
#            lp(dist)
            
            bmAbs += [dist]
            norm = meanComb #(y + meanComb) / 2
            norm = np.where(norm < 1, 1, norm)
            bmRel += [dist/norm]

        return np.mean(bmAbs, axis=0), np.mean(bmRel, axis=0)
#        return np.mean(bmAbs, axis=0), np.std(benchmark, axis=0)
    
    def aux_Ex_benchmark_bak(self, n, nMC, seed=None):
        """trains rgrEx separately
        
        n : int
            number of pairs
        nMC : int
            number of Monte Carlo trials to verify
        seed : int
            random numbers seed
        """
        benchmark = []
        for nn in range(n):
            diceOld = Dice()
            deciRr = np.random.choice([True, False], size=5)
            x = self.encode_Ex(diceOld, deciRr)
            y = self.rgrEx.predict(x.reshape(1, -1))[0]
            yMC = np.empty(shape=(nMC, 13))
            for mm in range(nMC):
                diceNew = diceOld.roll(deciRr)
                _, yMC[mm, :] = self.encode_Ex_xy(diceOld, deciRr, diceNew)
            yMC = np.mean(yMC, axis=0)
            benchmark += [np.linalg.norm(y-yMC)]
        return np.mean(benchmark), np.std(benchmark)
    
    def to_repMem(self, gameLog):
        """Add expereience to score regressor replay memory
        
        if possible in format:
        state1, action, reward, state2
        
        formats
        -------
        rgrScrCat: (sb, rew, sb)
        rgrDeltaRr: (sb, attempt, dice, deciRr, dice)
        rgrChances: (dice, deciRr, dice)
        
        gameLog : Game.log
        """
        for ii in range(13):
            sb1 = gameLog.loc[ii, 'scoreBoard']
            sb2 = gameLog.loc[ii+1, 'scoreBoard']
            reward = sb2.getSum() - sb1.getSum()
            self.repMemSC += [(sb1, reward, sb2)]
            
            dice0, deci0 = gameLog.loc[ii, ['dice0', 'deci0']]
            dice1, deci1 = gameLog.loc[ii, ['dice1', 'deci1']]
            dice2 = gameLog.loc[ii, 'dice2']
            self.repMemRr += [(sb1, 0, dice0, deci0, dice1)]
            self.repMemRr += [(sb1, 1, dice1, deci1, dice2)]
            
            self.repMemEx += [(dice0, deci0, dice1), (dice1, deci1, dice2)]
            
        self.repMemSC = self.repMemSC[-self.nGamesPreplayMem*13:]
        self.repMemRr = self.repMemRr[-self.nGamesPreplayMem*26:]
        self.repMemEx = self.repMemEx[-self.nGamesPreplayMem*26:]
    

    
    def train(self, nGames, benchmarkSeed,
              expl_cat_fct=explore_boltzmann,
              expl_cat_params={'minMaxRat': 5000},
              expl_rr_fct=explore_boltzmann,
              expl_rr_params={'minMaxRat': 50000},
              updTemp = 1.44
              ):
        """Training the Player with nGames and based on the trainers moves.
    
        Extended description of function.
    
        Parameters
        ----------
        nGames : int
            Nomber of games
        trainerEnsemble : PlayerEnsemble
            Integer represents the weight of the specici players moves
            player is someone doing decisions; None is self
        pRandCat : 0 <= float <= 1
            Probability for a random move (choose cat).
            Equivalent to epsilon of eplison-greedy startegy.
            Increase this factor in case of anti-learning (i.e. systematically
            decreaings benchmarks to a bad decisions on purpose startegy)
        pRat : float
            chose an option for training which promisses a
            high score. A weighted choose is performed where
            the best option is pRat times as likely as
            the worst option.
        updTemp : float
            Boltzman probability temperature for
            updating/reseting the rgrs after training
            1.44: 50% for decrease of 1 point
    
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
#        if self.nGames == 0:
#            # initiallize rgrEx with some random samples first
#            np.random.seed(self.nGames)
#            self.aux_Ex_train(n=100000)
        
#        assert 0 <= pOptRr + pRandRr <= 1
        for gg in range(nGames):
            np.random.seed(self.nGames)
            if self.nGames == 0:
                tmpPlayer = PlayerRandom()
            else:
                def fct_choose_reroll(scoreBoard, dice, attempt):
                    sb = scoreBoard
                    info = ''
                    
                    opts = self.eval_options_reroll(sb, dice, attempt)
                    actions = [opt[0] for opt in opts]
                    qs = [opt[1] for opt in opts]
                    reroll = expl_rr_fct(actions, qs, **expl_rr_params)
                    
                    return reroll, info
                
                def fct_choose_cat(scoreBoard, dice):
                    sb = scoreBoard
                    info = ''
                    
                    opts = self.eval_options_cat(sb, dice)
#                    opts = np.array(opts)
                    actions = [opt[0] for opt in opts]
                    qs = [opt[1] for opt in opts]
                    cat = expl_cat_fct(actions, qs, **expl_cat_params)
                    
                    return cat, info

                tmpPlayer = PlayerTemporary(
                        fct_choose_cat=fct_choose_cat,
                        fct_choose_reroll=fct_choose_reroll)

            game = Game(tmpPlayer)
            self.to_repMem(game.log)
            
            
            
            #rgrSC
            n_samples = self.nGamesPartFit * 13
            X = np.empty(shape=(n_samples, self.nFeat_SC))
            y = np.empty(shape=(n_samples,))
            allInds = list(range(len(self.repMemSC)))
            replace=False
            if len(allInds) < n_samples:
                replace = True
            inds = np.random.choice(allInds, size=n_samples, replace=replace)
            for nn, ind in enumerate(inds):
                X[nn, :], y[nn] = self.repMemSC_xy(ind)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.rgrSC = self.rgrSC.partial_fit(X, y)
#                self.rgrSC.partial_fit(X, y)
            
#            #rgrEx
#            n_samples = self.nGamesPartFit * 26
#            X = np.empty(shape=(n_samples, self.nFeat_Ex))
#            y = np.empty(shape=(n_samples, 13))
#            allInds = list(range(len(self.repMemEx)))
#            replace=False
#            if len(allInds) < n_samples:
#                replace = True
#            inds = np.random.choice(allInds, size=n_samples, replace=replace)
#            for nn, ind in enumerate(inds):
#                X[nn, :], y[nn, :] = self.repMemEx_xy(ind)
#            with warnings.catch_warnings():
#                warnings.filterwarnings("ignore", category=ConvergenceWarning)
#                self.rgrEx = self.rgrEx.partial_fit(X, y)
##                self.rgrEx.partial_fit(X, y)
            
            #rgrRr
            n_samples = self.nGamesPartFit * 26
            X = np.empty(shape=(n_samples, self.nFeat_Rr))
            y = np.empty(shape=(n_samples,))
            allInds = list(range(len(self.repMemRr)))
            replace=False
            if len(allInds) < n_samples:
                replace = True
            inds = np.random.choice(allInds, size=n_samples, replace=replace)
            for nn, ind in enumerate(inds):
                X[nn, :], y[nn] = self.repMemRr_xy(ind)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.rgrRr = self.rgrRr.partial_fit(X, y)
#                self.rgrRr.partial_fit(X, y)

            self.nGames += 1
            
#        # check if training brought some improovement
#        m, s = self.benchmark(seed=benchmarkSeed)
#        update = False
#        if self.rgrBaks is None or self.rgrBaks[0] < m:
#            update = True
#        else:
#            pUpdate = np.exp((m-self.rgrBaks[0])/updTemp)
#            if np.random.rand() <= pUpdate:
#                update = True
#        if update:
#            self.rgrBaks = [
#                    m, s, deepcopy(self.rgrSC),
#                    deepcopy(self.rgrEx), deepcopy(self.rgrRr)]
##            lp('updated')
#        else:
#            self.rgrSC = deepcopy(self.rgrBaks[2])
#            self.rgrEx = deepcopy(self.rgrBaks[3])
#            self.rgrRr = deepcopy(self.rgrBaks[4])
##            lp('rgrs reset', m, s, self.benchmark(seed=benchmarkSeed))
##        lp('experience: ', len(self.rgrSC.loss_curve_),
##           len(self.rgrRr.loss_curve_), len(self.rgrEx.loss_curve_))
    
    def save(self, filename):
        """Store this instance as pickle"""
        # delete loss curves to save filesize
        self.rgrSC.loss_curve_ = []
        self.rgrRr.loss_curve_ = []
        self.rgrEx.loss_curve_ = []
        
        pickle.dump(self, open(filename, "wb" ) )
    
    # https://stackoverflow.com/a/37658673
    def load(self, filename):
        try:
            player = pickle.load(open(filename, "rb"))
        except FileNotFoundError:
#            print('Coluld not loaded player! File:', filename, 'not found!')
            return
        else:
            self.__dict__.update(player.__dict__)
#            print('Loaded player from file:', filename)
            