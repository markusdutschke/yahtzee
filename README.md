# yahtzee

In this project an AI-player for the famous dice game Yahtzee / Kniffel is created,
which achieves a higher average score in 100 test games then the average human player.

I therefor used a Q-Learning approach with deep neural networks (scikit-learn, MLP Regressor) and a Softmax exploration strategy.
A documentation and scientific discussion can be found in `article_and_documentation/article.pdf`.

## Current status

This is still work in progress.
Currently super human performance with a score of 241.6 is achieved after 8000 training games.
The `master` branch of this project stores the latest 'productive' version.
The interested reader is also invited to have a look into the branch `workInProgrss`,
where the most recent status is stored.
The workInProgress-version is not necessarily working out of the box.

## Output

Start `main.py` to get the following output:

```
================================================================================
Lets first have a look at the final performance of the trained AI:
================================================================================

AI-Players created by Markus Dutschke:

	Description                                        avg. Score
	--------------------------------------------------------------------------------
	PlayerAI_full_v0:                                  200.1 +/- 36.5
	PlayerAI_full_v1:                                  235.4 +/- 39.4
	PlayerAI_full_v2:                                  241.6 +/- 40.7

A view benchmarks for comparison (check papers in README):

	Description                                        avg. Score
	--------------------------------------------------------------------------------
	PlayerAlg_oneShot_greedy:                          115.1 +/- 27.5
	PlayerAlg_full_greedy:                             170.7 +/- 28.8

	Random, no Bonus (Verhoff):                        46.0
	Greedy (Glenn06):                                  218.1
	Greedy (Felldin):                                  221.7
	Optimal Strategy (Holderied):                      245.9

	Human trials, group 1, 26 games:                   239.5
	Human trials, group 2, 8 games:                    202.8


================================================================================
Okay, lets check out a few games of the best AI Player!
================================================================================

==================================== Score Board ====================================
Category        : Score | round - dice (r = reroll)
-------------------------------------------------------------------------------------
Aces            : 2     |     5 - [1,1,4r,6,6] -> [1,1,5r,6,6] -> [1,1,5,6,6]
Twos            : 2     |     6 - [1r,1r,3,4,5] -> [1r,3r,4r,5,5] -> [1,1,2,5,5]
Threes          : 6     |     4 - [2,3,4,6r,6] -> [2,3,4r,4,6r] -> [1,2,3,3,4]
Fours           : 8     |     8 - [1r,1r,3r,3r,4] -> [2r,3r,4,4,6r] -> [2,3,4,4,5]
Fives           : 10    |     3 - [1r,1r,5,5,6r] -> [1r,2r,3r,5,5] -> [1,5,5,6,6]
Sixes           : 12    |     1 - [1r,5r,5r,6,6] -> [1r,2r,3r,6,6] -> [2,3,5,6,6]
-------------------------------------------------------------------------------------
Upper Sum       : 40    |
Bonus           : --    |
-------------------------------------------------------------------------------------
Three Of A Kind : 23    |     2 - [1r,2r,3r,6,6] -> [1r,1r,4r,6,6] -> [1,4,6,6,6]
Four Of A Kind  : 26    |    10 - [1r,1r,4r,5,5] -> [2r,3r,5,5,5] -> [5,5,5,5,6]
Full House      : 25    |     9 - [2,2,2,4r,5r] -> [2,2,2,5,5] -> [2,2,2,5,5]
Small Straight  : 30    |     0 - [1r,1,2,3,4] -> [1r,1,2,3,4] -> [1,2,3,3,4]
Large Straight  : 0     |    12 - [1r,1r,5r,5,6r] -> [1,2,4,5r,5] -> [1,2,2,4,5]
Yahtzee         : 50    |     7 - [1r,1r,2r,2r,5] -> [1r,3r,4r,5,5] -> [5,5,5,5,5]
Chance          : 27    |    11 - [1r,2r,5,5,6] -> [5,5,5,6,6] -> [5,5,5,6,6]
=====================================================================================
                                     Score:   221
=====================================================================================

==================================== Score Board ====================================
Category        : Score | round - dice (r = reroll)
-------------------------------------------------------------------------------------
Aces            : 4     |    12 - [1,1,2r,3r,4r] -> [1,1,1,1,2r] -> [1,1,1,1,6]
Twos            : 8     |     5 - [2,2,4r,5r,6r] -> [1r,2,2,3r,6r] -> [2,2,2,2,3]
Threes          : 12    |     3 - [1r,3,3,3,6r] -> [3,3,3,3,4r] -> [3,3,3,3,4]
Fours           : 8     |     9 - [1r,2r,3r,5r,6r] -> [1r,1r,4,5r,6r] -> [2,2,4,4,6]
Fives           : 20    |     1 - [1r,2r,3r,5,5] -> [4r,5,5,5,5] -> [5,5,5,5,6]
Sixes           : 12    |     6 - [3r,5r,5r,6,6] -> [2r,4,4,6,6] -> [2,4,4,6,6]
-------------------------------------------------------------------------------------
Upper Sum       : 64    |
Bonus           : 35    |
-------------------------------------------------------------------------------------
Three Of A Kind : 28    |     2 - [2r,5r,6,6,6] -> [1r,3r,6,6,6] -> [5,5,6,6,6]
Four Of A Kind  : 22    |    10 - [3r,4,4,4,6r] -> [4,4,4,4,6] -> [4,4,4,4,6]
Full House      : 25    |     4 - [1,1,2r,3r,4r] -> [1,1,1,2r,5r] -> [1,1,1,6,6]
Small Straight  : 30    |     8 - [2r,2r,4,5,6] -> [3,4,5r,5,6r] -> [3,3,4,5,6]
Large Straight  : 40    |     0 - [2,3r,3,4,5] -> [2,3r,3,4,5] -> [2,3,4,5,6]
Yahtzee         : 50    |     7 - [2r,3r,3r,5,5] -> [3r,3r,5,5,6r] -> [5,5,5,5,5]
Chance          : 26    |    11 - [1r,4,4,6,6] -> [4,4,6,6,6] -> [4,4,6,6,6]
=====================================================================================
                                     Score:   320
=====================================================================================

==================================== Score Board ====================================
Category        : Score | round - dice (r = reroll)
-------------------------------------------------------------------------------------
Aces            : 4     |     6 - [1,1,2r,3r,4r] -> [1,1,1,1,3r] -> [1,1,1,1,3]
Twos            : 2     |     8 - [1r,2,3r,4,5r] -> [1r,1r,2,4r,5r] -> [2,3,3,4,6]
Threes          : 9     |     1 - [1r,4r,5,5,6r] -> [1r,3,3,5r,5r] -> [1,3,3,3,5]
Fours           : 8     |    10 - [1r,1r,3r,5,6] -> [1r,2r,4,5r,6r] -> [2,3,4,4,5]
Fives           : 15    |     7 - [1r,2r,4r,5,5] -> [2r,3r,5,5,5] -> [1,2,5,5,5]
Sixes           : 18    |     0 - [1r,4r,5r,6,6] -> [2r,4r,5r,6,6] -> [1,2,6,6,6]
-------------------------------------------------------------------------------------
Upper Sum       : 56    |
Bonus           : --    |
-------------------------------------------------------------------------------------
Three Of A Kind : 22    |     3 - [3r,3r,5r,6,6] -> [1r,5r,6,6,6] -> [1,3,6,6,6]
Four Of A Kind  : 22    |    12 - [3r,3r,4,4,5r] -> [2r,3r,4,4,6r] -> [4,4,4,4,6]
Full House      : 25    |     2 - [1r,2r,4,4,6r] -> [2,2,4,4,4] -> [2,2,4,4,4]
Small Straight  : 30    |     5 - [1r,3,4,5,6r] -> [3r,3,4,5,6] -> [3,4,4,5,6]
Large Straight  : 40    |     4 - [1r,4,5r,5,6r] -> [1r,1r,4r,4,5] -> [2,3,4,5,6]
Yahtzee         : 0     |     9 - [1r,2r,2r,4,6r] -> [1r,1r,1r,2r,4] -> [2,3,4,4,6]
Chance          : 17    |    11 - [2r,2r,4,4,6] -> [2r,4,4,5r,6] -> [1,2,4,4,6]
=====================================================================================
                                     Score:   212
=====================================================================================



================================================================================
Now, let's see how such a cool AI player is trained ...
================================================================================

Note: training + benchmarks takes a few hours

	# Trainings          Score
	1                    62.3 +/- 21.5
	2                    67.7 +/- 23.4
	3                    72.3 +/- 25.2
	4                    84.3 +/- 25.5
	5                    103.9 +/- 24.2
	6                    115.9 +/- 33.4
	7                    128.7 +/- 35.4
	8                    157.1 +/- 35.9
	9                    185.0 +/- 34.4
	10                   192.8 +/- 34.9
	20                   203.9 +/- 38.2
	30                   202.6 +/- 35.9
	40                   201.7 +/- 42.3
	50                   202.5 +/- 38.7
	60                   179.2 +/- 29.7
	70                   187.4 +/- 27.4
	80                   187.5 +/- 26.9
	90                   191.2 +/- 28.4
	100                  194.0 +/- 29.3
	200                  214.5 +/- 31.8
	300                  214.6 +/- 37.5
	400                  218.2 +/- 36.6
	500                  217.8 +/- 32.5
	600                  218.9 +/- 34.9
	700                  221.2 +/- 37.1
	800                  219.5 +/- 37.1
	900                  215.7 +/- 32.4
	1000                 215.2 +/- 34.2
	1100                 226.4 +/- 39.2
	1200                 224.2 +/- 36.5
	1300                 224.4 +/- 36.3
	1400                 225.1 +/- 33.6
	1500                 220.5 +/- 35.8
	1600                 222.4 +/- 34.4
	1700                 225.9 +/- 37.9
	1800                 222.4 +/- 35.3
	1900                 218.8 +/- 32.7
	2000                 219.8 +/- 34.4
	2100                 222.2 +/- 33.0
	2200                 225.4 +/- 37.4
	2300                 226.1 +/- 36.3
	2400                 223.1 +/- 39.8
	2500                 224.8 +/- 35.8
	2600                 223.2 +/- 38.0
	2700                 226.9 +/- 37.7
	2800                 219.7 +/- 37.0
	2900                 226.1 +/- 38.2
	3000                 226.8 +/- 36.5
	3100                 231.1 +/- 36.2
	3200                 232.6 +/- 40.4
	3300                 226.1 +/- 42.2
	3400                 220.5 +/- 37.7
	3500                 223.1 +/- 37.5
	3600                 222.2 +/- 41.3
	3700                 221.9 +/- 35.8
	3800                 222.4 +/- 34.4
	3900                 224.3 +/- 39.8
	4000                 230.0 +/- 41.8
	4100                 222.9 +/- 37.6
	4200                 228.6 +/- 38.0
	4300                 225.6 +/- 38.6
	4400                 217.9 +/- 32.5
	4500                 227.6 +/- 41.9
	4600                 224.2 +/- 43.2
	4700                 232.1 +/- 39.0
	4800                 229.0 +/- 38.1
	4900                 229.0 +/- 36.6
	5000                 229.5 +/- 37.7
	5100                 227.1 +/- 38.0
	5200                 228.9 +/- 34.0
	5300                 226.7 +/- 36.1
	5400                 222.2 +/- 35.1
	5500                 225.9 +/- 41.4
	5600                 229.2 +/- 43.8
	5700                 229.4 +/- 42.7
	5800                 231.8 +/- 43.6
	5900                 230.9 +/- 42.9
	6000                 228.2 +/- 42.0
	6100                 229.2 +/- 42.2
	6200                 226.5 +/- 40.4
	6300                 223.0 +/- 40.4
	6400                 229.8 +/- 38.0
	6500                 234.5 +/- 41.0
	6600                 228.0 +/- 40.6
	6700                 234.2 +/- 46.1
	6800                 231.6 +/- 40.2
	6900                 233.1 +/- 40.1
	7000                 224.7 +/- 36.8
	7100                 228.4 +/- 39.1
	7200                 230.6 +/- 42.3
	7300                 230.3 +/- 38.1
	7400                 231.2 +/- 42.9
	7500                 228.4 +/- 42.3
	7600                 231.7 +/- 43.3
	7700                 235.1 +/- 41.5
	7800                 229.3 +/- 42.9
	7900                 232.1 +/- 42.5
	8000                 241.6 +/- 40.7
```



# Further material

## probabilities and strategies
- http://www.brefeld.homepage.t-online.de/kniffel.html (German)
- https://en.wikipedia.org/wiki/Yahtzee#Optimal_strategy

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
