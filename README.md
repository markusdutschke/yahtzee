# yahtzee

Call
$ python main.py
to train and test an artificially intelligent player for the dice game yahtzee.

## Output


```
================================================================================
Lets first have a look at the final performance of the trained AI:
================================================================================

AI-Players created by Markus Dutschke:

        Description                                        avg. Score
        --------------------------------------------------------------------------------
        PlayerAI_full_v0:                                  200.1 +/- 18.7
        PlayerAI_full_v1:                                  216.7 +/- 17.7

A view benchmarks for comparison (check papers in README):

        Description                                        avg. Score
        --------------------------------------------------------------------------------
        Random, no Bonus (Verhoff):                        46.0
        Greedy strategy, no re-rolls:                      111.0
        Greedy (Glenn06):                                  218.1
        Greedy (Felldin):                                  221.7
        Optimal Strategy (Holderied):                      245.9
        Experienced human trials (self produced):          239.5


================================================================================
Okay, lets check out a few games of the best AI Player!
================================================================================

==================================== Score Board ====================================
Category        : Score | round - dice (r = reroll)
-------------------------------------------------------------------------------------
Aces            : 2     |    11 - [2r,2r,5r,5r,6r] -> [1,1,4r,4r,6r] -> [1,1,3,6,6]
Twos            : 4     |    10 - [2,2,3r,6r,6r] -> [1,1,2,2,6r] -> [1,1,2,2,4]
Threes          : 9     |     9 - [1r,3,3,5r,5r] -> [1r,2r,3,3,6r] -> [2,3,3,3,6]
Fours           : 8     |     5 - [1r,1r,2r,3,4] -> [2r,3,3,4,4] -> [1,3,3,4,4]
Fives           : 20    |     7 - [4r,5,5,5,6] -> [5,5,5,5,6r] -> [2,5,5,5,5]
Sixes           : 24    |     6 - [1r,1r,4r,6,6] -> [3r,5r,6,6,6] -> [4,6,6,6,6]
-------------------------------------------------------------------------------------
Upper Sum       : 67    |
Bonus           : 35    |
-------------------------------------------------------------------------------------
Three Of A Kind : 25    |     0 - [1,2,4,5,6r] -> [1r,2r,2r,4,5] -> [4,5,5,5,6]
Four Of A Kind  : 14    |     2 - [2,2,2,2,6r] -> [1r,2,2,2,2] -> [2,2,2,2,6]
Full House      : 25    |     4 - [1r,2r,2r,3,5r] -> [2,2,3,3,3] -> [2,2,3,3,3]
Small Straight  : 30    |     3 - [2r,2,4r,4,6r] -> [2r,2r,4,5r,5] -> [3,4,5,5,6]
Large Straight  : 40    |     1 - [3,4,5r,5,6] -> [3,4,5r,5,6] -> [2,3,4,5,6]
Yahtzee         : 0     |    12 - [1,1,5r,6r,6r] -> [1,1,2,2,6r] -> [1,1,2,2,3]
Chance          : 19    |     8 - [1r,2r,3,5,5] -> [1r,3r,5,5,5] -> [1,3,5,5,5]
=====================================================================================
                                     Score:   255
=====================================================================================

==================================== Score Board ====================================
Category        : Score | round - dice (r = reroll)
-------------------------------------------------------------------------------------
Aces            : 1     |    11 - [1r,3,4r,5,5] -> [3,4r,4r,5,5] -> [1,3,5,5,5]
Twos            : 2     |    10 - [3,5r,6,6,6] -> [3r,4,6,6,6] -> [2,4,6,6,6]
Threes          : 6     |     2 - [1r,1r,3,4r,6] -> [1r,2,3,5,6] -> [2,3,3,5,6]
Fours           : 8     |     6 - [1r,4,4,5,5] -> [1r,4,4,5,5] -> [3,4,4,5,5]
Fives           : 10    |     9 - [2r,3,4r,5,5] -> [3r,4r,5,5,6r] -> [1,5,5,6,6]
Sixes           : 12    |     4 - [1r,2,3,4,6] -> [2,3,4,6r,6] -> [2,3,4,6,6]
-------------------------------------------------------------------------------------
Upper Sum       : 39    |
Bonus           : --    |
-------------------------------------------------------------------------------------
Three Of A Kind : 16    |     0 - [1r,1r,2r,3,6] -> [2r,3,3,3,6r] -> [2,3,3,3,5]
Four Of A Kind  : 11    |     3 - [1r,2,2,2,2] -> [2,2,2,2,4r] -> [2,2,2,2,3]
Full House      : 0     |    12 - [1r,2r,3,3,5] -> [1r,2r,3,3,5] -> [3,3,3,4,5]
Small Straight  : 30    |     1 - [1r,3,4,5,6] -> [3,4,5,6r,6] -> [3,3,4,5,6]
Large Straight  : 40    |     8 - [1,2r,2,4,5] -> [1,2,3,4,5] -> [1,2,3,4,5]
Yahtzee         : 0     |     7 - [1r,1,3,4,6r] -> [1r,3,4,5,6] -> [1,3,4,5,6]
Chance          : 24    |     5 - [1r,3,4,5,6] -> [3,4r,4,5,6] -> [3,4,5,6,6]
=====================================================================================
                                     Score:   160
=====================================================================================

==================================== Score Board ====================================
Category        : Score | round - dice (r = reroll)
-------------------------------------------------------------------------------------
Aces            : 3     |    10 - [2r,2r,4r,4r,5r] -> [1,1,2r,2r,6r] -> [1,1,1,2,6]
Twos            : 6     |     8 - [2,2,2,3,4r] -> [2,2,2,3,6r] -> [2,2,2,3,6]
Threes          : 6     |    11 - [1r,4r,4r,5r,5r] -> [2r,3,3,4r,6r] -> [1,3,3,5,6]
Fours           : 8     |     3 - [2r,2,4r,4,6r] -> [2r,4,4,5r,5r] -> [1,3,4,4,5]
Fives           : 20    |     7 - [2r,2r,4r,5,6r] -> [1r,5,5,5,6r] -> [2,5,5,5,5]
Sixes           : 18    |     2 - [1r,2r,5r,6,6] -> [2r,5r,5r,6,6] -> [2,5,6,6,6]
-------------------------------------------------------------------------------------
Upper Sum       : 61    |
Bonus           : --    |
-------------------------------------------------------------------------------------
Three Of A Kind : 25    |     1 - [1r,2r,2r,4r,5] -> [2r,5r,5r,6,6] -> [3,4,6,6,6]
Four Of A Kind  : 18    |     4 - [1r,3,3,3,5r] -> [3,3,3,4r,6r] -> [3,3,3,3,6]
Full House      : 25    |     9 - [1r,3,4r,6r,6r] -> [3,4r,6r,6,6] -> [3,3,6,6,6]
Small Straight  : 30    |     6 - [2,3,4,5,6] -> [2,3,4,5,6] -> [2,3,4,5,6]
Large Straight  : 40    |     0 - [1r,3,4,6r,6r] -> [1r,3r,3,4,5] -> [2,3,4,5,6]
Yahtzee         : 50    |    12 - [3,3,3,6r,6r] -> [3,3,3,3,5r] -> [3,3,3,3,3]
Chance          : 21    |     5 - [1r,2r,2r,4,5] -> [3r,3,4,5r,5] -> [3,4,4,5,5]
=====================================================================================
                                     Score:   270
=====================================================================================



================================================================================
Now, let's see how such a cool AI player is trained ...
================================================================================

Note: training + benchmarks takes a few hours

        # Trainings          Score
        1                    168.6 +/- 13.5
        2                    176.9 +/- 14.0
        3                    176.2 +/- 15.3
        4                    179.4 +/- 19.3
        10                   186.8 +/- 19.3
        20                   181.7 +/- 13.3
        30                   182.5 +/- 11.6
        40                   180.8 +/- 12.4
        50                   182.9 +/- 14.1
        60                   193.6 +/- 15.8
        70                   194.4 +/- 13.2
        80                   197.8 +/- 13.9
        90                   202.0 +/- 15.8
        100                  202.9 +/- 12.6
        200                  211.3 +/- 15.3
        300                  214.8 +/- 18.2
        400                  216.3 +/- 19.1
        500                  219.3 +/- 14.2
        600                  217.2 +/- 13.6
        700                  212.7 +/- 18.2
        800                  218.9 +/- 16.4
        900                  216.3 +/- 14.0
```



# Further material

## probabilities for some combination
http://www.brefeld.homepage.t-online.de/kniffel.html (German)

## simulator
- http://yahtzee.holderied.de/
- http://kniffel.holderied.de/ (German)


## references (yahtzee)
- Holderied: http://holderied.de/kniffel/ (same rules)
- Glenn06: http://gunpowder.cs.loyola.edu/~jglenn/research/optimal_yahtzee.pdf (other rules)
- Jedenberg: https://www.diva-portal.org/smash/get/diva2:810580/FULLTEXT01.pdf (other rules)
- Verhoff: http://www.yahtzee.org.uk/optimal_yahtzee_TV.pdf
- Felldin: http://www.csc.kth.se/utbildning/kth/kurser/DD143X/dkand12/Group5Mikael/final/Markus_Felldin_and_Vinit_Sood.pdf (same rules)

## references Q-Learning
- Mnih13: https://arxiv.org/abs/1312.5602
- Tijsma16: https://www.researchgate.net/profile/Marco_Wiering/publication/311486379_Comparing_Exploration_Strategies_for_Q-learning_in_Random_Stochastic_Mazes/links/5a96639b45851535bcdccdda/Comparing-Exploration-Strategies-for-Q-learning-in-Random-Stochastic-Mazes.pdf?origin=publication_detail
