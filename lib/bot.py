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
#np.random.seed(0)
from yahtzee import Game, ScoreBoard
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
            regressor=MLPRegressor(hidden_layer_sizes=(30, 25, 20)),
            debugLevel=0,
            catMLParas={'lenReplayMem': 1000, 'lenMiniBatch': 10, 'gamma': .1}
#            catReplayMemLen=1000,
#            catMiniBatchSize=10,
            
#            playerInit=PlayerRandomCrap(),
#            nGamesInit=1
            ):
        super().__init__()
        self.rgr = regressor
        self.nGames = 0
        self.debugLevel = debugLevel
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
    
    def eval_cats(self, scoreBoard, dice):
        """retuns list of (cat nr, predicted score)
        First element has highest score"""
        opts = []
        for cat in scoreBoard.open_cats():
            score = self.cat_predict(scoreBoard, dice, cat)
            assert len(score)==1
            score = score[0]
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
        self.crm = self.crm[-self.catMLParas['lenReplayMem']:]
    
    def xy_from_crm(self, crmElem):
        assert len(crmElem) in [4,6]
        sb1, dice1, cat1, sc = crmElem[:4]
        
        x = self.encoder(sb1, dice1, cat1)
        y = sc
        if len(crmElem) == 6 and self.nGames > 0:
            sb2, dice2 = crmElem[4:]
            opts = self.eval_cats(sb2, dice2)
            y += self.catMLParas['gamma'] * opts[0][1]
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
                xy = self.xy_from_crm(crmElem)
                X[ind, :] = xy[0]
                y[ind] = xy[1]

#            scoreBoards, dices, cats = self.games_to_cat_info(game)
#            lp(scoreBoards)
#            lp(dices)
#            lp(cats)
#            X, y = self.cat_decision_parser(scoreBoards, dices, cats)

            self.rgr.partial_fit(X, y)
            self.nGames += 1


class PlayerAI_1SEnc_1(PlayerOneShotAI):
    name = 'AI_1SEnc_1'
    
    def __init__(
            self, regressor=MLPRegressor(hidden_layer_sizes=(20, 15, 10, 5))):
        super().__init__(regressor)
    
    
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
        assert cat in scoreBoard.open_cats()  # put to LearningPlayer and call by super()
        x = np.zeros(shape=(self.n_features))

#        for cc in scoreBoard.open_cats():
#            x[cc] = scoreBoard.check_points(dice, cc) / 50
#        x[-4:] = np.array(list(np.binary_repr(cat, width=4)))
        if cat in scoreBoard.open_cats():
            x[cat] = scoreBoard.check_points(dice, cat) # / 50
        
        return x

class PlayerAI_1SEnc_2(PlayerOneShotAI):
    name = 'AI_1SEnc_2'
    
    def __init__(
            self, regressor=MLPRegressor(hidden_layer_sizes=(20, 15, 10, 5))):
        super().__init__(regressor)
        
    
    
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

#        for cc in scoreBoard.open_cats():
#            x[cc] = scoreBoard.check_points(dice, cc) / 50
#        x[-4:] = np.array(list(np.binary_repr(cat, width=4)))
#        if cat in scoreBoard.open_cats():
        x[cat] = scoreBoard.check_points(dice, cat) / 50
        
        return x

class PlayerAI_1SEnc_3(PlayerOneShotAI):
    name = 'AI_1SEnc_3'
    
    def __init__(
            self, regressor=MLPRegressor(hidden_layer_sizes=(20, 15, 10, 5),
                                         )):
        super().__init__(regressor)
#        lp(self.rgr.get_params())
    
    
    @property
    def n_features(self):
        """size or regressor input, reffers to MLPRegressor.fit
        Directly coupled to self.encoder.
        """
        return 13 + 13 + 1
    def encoder(self, scoreBoard, dice, cat):
        """Encodes a game situation (decision input) as
        array with elements in range 0 to 1.
        """
        assert 0 <= cat <= 12
        x = np.zeros(shape=(self.n_features))

#        for cc in scoreBoard.open_cats():
#            x[cc] = scoreBoard.check_points(dice, cc) / 50
#        x[-4:] = np.array(list(np.binary_repr(cat, width=4)))
#        if cat in scoreBoard.open_cats():
        x[cat] = scoreBoard.check_points(dice, cat) / 50
        x[13:26] = scoreBoard.scores.mask.astype(int)  # avail. cats
        x[-1] = scoreBoard.getSum()
        return x