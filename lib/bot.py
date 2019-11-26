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
#np.random.seed(0)
from yahtzee import Game, ScoreBoard, Dice
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_is_fitted
from comfct.list import list_cast
from comfct.numpy import weighted_choice
#from tqdm import tqdm
from progressbar import progressbar
#from progress.bar import Bar

#def benchmark(player, nGames=100):
#    """Benchmarks Yahtzee decision making models.
#    
#    Extended description of function.
#    
#    Parameters
#    ----------
#    fctRoll : function
#        Classifier for which dice to reroll.
#        arg1 : ScoreBoard
#        arg2 : Dice
#        arg3 : int, 0 or 1
#            number of reroll attempts so far
#        returns : bool array of len 5
#    fctCat : str
#        Classifier for which categorie to use on the score board for dice.
#        arg1 : ScoreBoard
#        arg2 : Dice
#        returns : 0 <= int <= 12
#    player : Player
#        Artificial player, subclass of AbstractPlayer
#    nGames : int
#        number of trials
#    
#    Returns
#    -------
#    scoreMean : float
#        Description of return value
#    scoreStd : float
#        Standard deviation
#    
#    See Also
#    --------
#    otherfunc : some related other function
#    
#    Examples
#    --------
#    These are written in doctest format, and should illustrate how to
#    use the function.
#    
#    >>> a=[1,2,3]
#    >>> [x + 3 for x in a]
#    [4, 5, 6]
#    """
#    scores = []
#    for ii in range(nGames):
#        game = Game(player)
#        scores += [game.sb.getSum()]
#    return np.mean(scores), np.std(scores)
    

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
    def choose_roll(self, scoreBoard, dice, attempt):
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
    
    def benchmark(self, nGames=100, nBins=20):
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
        scores = []
        for ii in range(nGames):
            game = Game(self)
            scores += [game.sb.getSum()]
#        lp(scores)
        scores = [np.mean(chunk) for chunk in np.array_split(scores, nBins)]
#        lp(scores)
        return np.mean(scores), np.std(scores)/len(scores)**.5
    
class PlayerRandomCrap(AbstractPlayer):
    """This player behaves completely random"""
    name = 'Random Crap'
    def choose_roll(self, scoreBoard, dice, attempt):
        return np.random.choice([True, False], 5)
    def choose_cat(self, scoreBoard, dice):
        return np.random.choice(scoreBoard.open_cats())

class PlayerOneShotHero(AbstractPlayer):
    """This player assigns the dice to the category with the most scores"""
    name = 'Mr. One Shot Hero'
    def choose_roll(self, scoreBoard, dice, attempt):
        return [False]*5
    def choose_cat(self, scoreBoard, dice):
#        print(151, dice, type(dice))
        bench = []
        for cat in scoreBoard.open_cats():
            bench += [(scoreBoard.check_points(dice, cat), cat)]
#        lp(bench)
        bench = sorted(bench, key=lambda x: x[0])
        return bench[-1][1]

class PlayerOneShotAI_deprictaed(AbstractPlayer):
    """No strategic dice reroll, but self learning category assignment"""
    name = 'The One Shot AI'

    def __init__(
            self,
            regressor=MLPRegressor(hidden_layer_sizes=(30, 25, 20)),
            debugLevel=0,
#            playerInit=PlayerRandomCrap(),
#            nGamesInit=1
            ):
        super().__init__()
        self.rgr = regressor
        self.nGames = 0
        self.debugLevel = debugLevel
        
        # init regressor for adaptative fit
#        games = [Game(playerInit) for ii in range(nGamesInit)]
#        X, y = self.cat_decision_parser(*self.games_to_cat_info(games))
##        self.rgr.fit(X, y)
#        self.rgr.partial_fit(X, y)
        
    def choose_roll(self, scoreBoard, dice, attempt):
        return [False]*5
    
    # move to abstract LearningPlayer
    def choose_cat(self, scoreBoard, dice):
        """Takes only open categories."""
        assert self.nGames > 0, str(scoreBoard.print())
        if self.debugLevel > 0:
            print('='*60)
            print('='*60)
            scoreBoard.print()
            print('DICE: ', dice)
            
        opts = []
        for cat in scoreBoard.open_cats():
            score = self.cat_predict(scoreBoard, dice, cat)
            assert len(score)==1
            score = score[0]
            opts += [(cat, score)]
            if self.debugLevel > 0:
                print('cat: {:}, score: {:.2f}'.format(
                        ScoreBoard.cats[cat], score))
#                print(323, ScoreBoard.cats[cat], score)
        opts = sorted(opts, key=lambda x: x[1])
        if self.debugLevel > 0:
            print('-> chose cat:',ScoreBoard.cats[opts[-1][0]])
        return opts[-1][0]
#        cs = self.cat_predict(scoreBoard, dice)
##        cs = self.rgr.predict(self.cat_decision_parser(scoreBoard, dice))
#        cs = np.ma.masked_array(cs, mask=np.invert(scoreBoard.mask))
#        return np.argmax(cs)
    
    def cat_predict(self, scoreBoard, dice, cat):
        X, y = self.cat_decision_parser(scoreBoard, dice, cat)
#        lp(X)
        return self.rgr.predict(X)
    
    def games_to_cat_info(self, games):
        """Extracts relevant information for category prediction
        from the games log
        """
        games = list_cast(games)
        scoreBoards = [game.log[rr][0] for game in games for rr in range(13)]
        dices = [game.log[rr][-2] for game in games for rr in range(13)]
        cats = [game.log[rr][-1] for game in games for rr in range(13)]
        return scoreBoards, dices, cats
    
    def cat_decision_parser(self, scoreBoards, dices, cats):
        """Prepares X, y tupes for regressor fit
        based on relevant information from games_to_cat_info
        cat : int
        """
        scoreBoards = list_cast(scoreBoards)
        dices = list_cast(dices)
        cats = list_cast(cats)
        assert len(scoreBoards) == len(dices) == len(cats)
        n_samples = len(scoreBoards)
#        lp(n_samples, self.n_features)
#        a = np.empty(shape=(13,31))
        X = np.empty(shape=(n_samples, self.n_features))
        y = np.empty(shape=(n_samples,))
        for ind in range(n_samples):
#        for ind, (scoreBoard, dice) in enumerate(zip(scoreBoards, dices)):
#            gs = game.score
#            for rr, roundLog in enumerate(games.log):
#                ind = gg*13 + rr
#            scoreBoard.print()
            scoreBoard = scoreBoards[ind]
            dice = dices[ind]
            cat = cats[ind]
            X[ind, :] = self.encoder(scoreBoard, dice, cat)
            y[ind] = scoreBoard.score
        return X, y
    
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 13 + 5 + 1
    def encoder(self, scoreBoard, dice, cat):
        """Encodes a game situation (decision input) as
        array with elements in range 0 to 1.
        """
        x = np.zeros(shape=(self.n_features))
#        lp(scoreBoard.scores.data)
        x[:13] = scoreBoard.scores.data / 50  # scores
        x[13:26] = scoreBoard.scores.mask.astype(int)  # avail. cats
        x[26:31] = (dice.vals -1) / 5  # dice
        x[31] = cat
        return x
        # hopefully learns bonus by itsself
#        x[0, 31] = np.sum(roundLog[0].data[:6]) / 105  # upper sum for bonus
    
    def train(self, nGames,
              trainerEnsemble=PlayerEnsemble(
                      [(1, PlayerRandomCrap())]
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
                    players[0] = PlayerRandomCrap()
#            lp(type(players), len(players))
            game = Game(players)
#            for rl in game.log:
#                rl[0].print()
#                lp(rl[-2], 'cat', rl[-1])
#            lp(self.name, self.n_features)
            X, y = self.cat_decision_parser(*self.games_to_cat_info(game))
#            lp(X.shape, y.shape)
            self.rgr.partial_fit(X, y)
            self.nGames += 1

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
            
#            playerInit=PlayerRandomCrap(),
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
        
    def choose_roll(self, scoreBoard, dice, attempt):
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
                    res += [sb.check_points(dice, cc)]
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
        x[cat] = scoreBoard.check_points(dice, cat)
##        lp(scoreBoard.scores.data)
#        x[:13] = scoreBoard.scores.data / 50  # scores
#        x[13:26] = scoreBoard.scores.mask.astype(int)  # avail. cats
#        x[26:31] = (dice.vals -1) / 5  # dice
#        x[31] = cat
        return x
        # hopefully learns bonus by itsself
#        x[0, 31] = np.sum(roundLog[0].data[:6]) / 105  # upper sum for bonus
    
    def train(self, nGames,
              trainerEnsemble=PlayerEnsemble(
                      [(1, PlayerRandomCrap())]
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
                    players[0] = PlayerRandomCrap()
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
            
    def train2(self, nGames, pRand=.1, pRat=100):
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
                        game.perf_action(act, self.choose_roll(*paras))
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
#                    players[0] = PlayerRandomCrap()
#            players = 0
#            if self.nGames == 0:
#                player = PlayerRandomCrap()
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

            self.rgr.partial_fit(X, y)
            self.cat_replay_memory_trunc()
            self.nGames += 1
    
    @classmethod
    def modelBenchmark(cls, nGames=range(1,2), nInstances=50, *args, **kwargs):
        np.random.seed(0)
        df = pd.DataFrame(index=nGames)
        df.index.name = 'nGames'
#        print('inst', flush=True, end='; ')
        for ii in progressbar(range(nInstances), 'Instances'):
#            print(ii, flush=True, end='; ')
            player = cls(*args, **kwargs)
            means = []
            stds = []
            for gg in nGames:
#                lp('game', gg)
                assert gg > 0, 'number of games <= 0 makes no sense'
                player.train2(nGames=gg-player.nGames)
                m, s = player.benchmark(nGames=50, nBins=10)
                means += [m]
                stds += [s]
            df['inst_'+str(ii)] = means
#        print()
        res = pd.DataFrame()
        res['mean'] = df.mean(axis=1)
        res['sem'] = (df.std(axis=1)**2 + np.array(stds)**2)**.5 / nInstances**.5
        res['max'] = df.max(axis=1)
        res['min'] = df.min(axis=1)
        return res #, df



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
        
        x[0] = scoreBoard.check_points(dice, cat)
        
        for cc in scoreBoard.open_cats():
            if cc==cat:
                continue  # helps converging
            x[1+cc] = scoreBoard.check_points(dice, cc)
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
        
        x[0] = scoreBoard.check_points(dice, cat)
        x[1] = scoreBoard.check_points(dice, cat) / scoreBoard.check_points_max(cat)
        
        for cc in scoreBoard.open_cats():
            if cc==cat:
                continue
            x[2+cc] = scoreBoard.check_points(dice, cc) / scoreBoard.check_points_max(cc)
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
        
        x[0] = scoreBoard.check_points(dice, cat)
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
        
        x[cat] = scoreBoard.check_points(dice, cat)
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
        
        x[cat] = scoreBoard.check_points(dice, cat)
        
        for cc in scoreBoard.open_cats():
            if cc==cat:
                continue  # helps converging
            x[13+cc] = scoreBoard.check_points(dice, cc)
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
        
        x[cat] = scoreBoard.check_points(dice, cat)

        return x

class PlayerOneShotAI_new(AbstractPlayer):
    """New AI Player concept.
    
    Regressor scrRgr focasts the rest score of the game
    based on the empty categories of a score board.
    
    Categorie decision is made based on the direct reward 
    + the reward forcast of the resulting scoreboard.
    """
    
    def __init__(
            self, mlpRgrArgs={'hidden_layer_sizes':(30, 25, 20)},
            lenScrReplayMem=200, lenMiniBatch=30, gamma=.98):
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
        self.srm = []
        self.lenScrReplayMem = lenScrReplayMem
        self.lenMiniBatch = lenMiniBatch
        self.gamma = gamma
        self.nGames = 0
        
    def choose_roll(self, scoreBoard, dice, attempt):
        return [False]*5
    
    def choose_cat(self, scoreBoard, dice, debugLevel=0):
        opts = self.eval_options_cat(scoreBoard, dice)
        return opts[0][0]
    
    def eval_options_cat(self, scoreBoard, dice):
        """Return a sorted list with the options to choose for cat and
        the expected restScore.
        """
        opts = []
        for cat in scoreBoard.open_cats():
            directReward = scoreBoard.check_points(dice, cat)
#            score = self.cat_predict(scoreBoard, dice, cat)
#            assert len(score)==1
#            score = score[0]
            
            # additionally consider the resulting state of the score board
            tmpSB = scoreBoard.copy()
            tmpSB.add(dice, cat)
#            score += self.catMLParas['gamma'] * self.predict_score(tmpSB)
            x = self.encode_scrRgr_x(scoreBoard).reshape(1, -1)
            futureReward = self.srcRgr.predict(x)
            
            reward = directReward + futureReward
            opts += [(cat, reward)]
        opts = sorted(opts, key=lambda x: x[1], reverse=True)
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
        return x
    
    def add_srm_sample(self, scoreBoard, restScore):
        """Save memory in format x, y"""
        x = self.encode_scrRgr_x(scoreBoard)
        y = restScore
        self.srm += [(x, y)]
    
    def truncate_srm(self):
        """Reduce srm length to lenScrReplayMem to the most recent memories.
        """
        self.srm = self.srm[-self.lenScrReplayMem:]
    
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
        for gg in range(nGames):
            game = Game()
            sbs = []
            for rr in range(13):
                for aa in range(3):
                    sbs += [game.sb]
                    act, paras = game.ask_action()
                    if aa < 2:
                        sb, dice, attempt = paras
                        game.perf_action(act,
                                         self.choose_roll(sb, dice, attempt))
                    else:
                        sb, dice = paras
                        if self.nGames == 0 or np.random.rand() <= pRand:
                            cat = np.random.choice(sb.open_cats())
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
            sbs += [game.sb]
            finalScore = game.sb.getSum()
            for sb in sbs:
                self.add_srm_sample(sb, finalScore-sb.getSum())

            #create miniBatch
            n_samples = self.lenMiniBatch
            X = np.empty(shape=(n_samples, self.n_features))
            y = np.empty(shape=(n_samples,))
            for ind in range(n_samples):
                xy = np.random.choice(self.srm)
                X[ind, :] = xy[0]
                y[ind] = xy[1]

            self.scrRgr.partial_fit(X, y)
            self.truncate_srm()
            self.nGames += 1
    
    @classmethod
    def modelBenchmark(cls, nGames=range(1,2), nInstances=50, *args, **kwargs):
        np.random.seed(0)
        df = pd.DataFrame(index=nGames)
        df.index.name = 'nGames'
#        print('inst', flush=True, end='; ')
        for ii in progressbar(range(nInstances), 'Instances'):
#            print(ii, flush=True, end='; ')
            player = cls(*args, **kwargs)
            means = []
            stds = []
            for gg in nGames:
#                lp('game', gg)
                assert gg > 0, 'number of games <= 0 makes no sense'
                player.train(nGames=gg-player.nGames)
                m, s = player.benchmark(nGames=50, nBins=10)
                means += [m]
                stds += [s]
            df['inst_'+str(ii)] = means
#        print()
        res = pd.DataFrame()
        res['mean'] = df.mean(axis=1)
        res['sem'] = (df.std(axis=1)**2 + np.array(stds)**2)**.5 / nInstances**.5
        res['max'] = df.max(axis=1)
        res['min'] = df.min(axis=1)
        return res #, df
    
    
        