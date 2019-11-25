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
from yahtzee import Game
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_is_fitted
from comfct.list import list_cast

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
#        probsSum = np.sum(
#        self.probs = [pr/np.sum(
    
    def rand(self):
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
        return random.choices(ps, ws, k=13*3)

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
    
    def benchmark(self, nGames=100, nBins=10):
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
        return np.mean(scores), np.std(scores)
    
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

class PlayerOneShotAI(AbstractPlayer):
    """No strategic dice reroll, but self learning category assignment"""
    name = 'The One Shot AI'

    def __init__(
            self,
            regressor=MLPRegressor(hidden_layer_sizes=(30, 25, 20)),
            playerInit=PlayerRandomCrap(),
            nGamesInit=1
            ):
        super().__init__()
        self.rgr = regressor
        self.nGames = 0
        
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
        opts = []
        for cat in scoreBoard.open_cats():
            opts += [(cat, self.cat_predict(scoreBoard, dice, cat))]
        opts = sorted(opts, key=lambda x: x[1])
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
            self.rgr.partial_fit(
                    *self.cat_decision_parser(*self.games_to_cat_info(game)))
            self.nGames += 1