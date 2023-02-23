# Tennis
This is our tennis database, covering all men's tennis matches from 2010 till now in the matches.csv table. The official senior level tournaments (mainly ATP Tour, ITF Circuit and Grand Slams) are taken from Jeff Sackman's dataset (https://github.com/JeffSackmann/tennis_atp), the rest of the data (e.g. junior tournaments, European league system, college tennis) is from various other sources. By merging the matches.csv table with the tournaments.csv, players.csv or cities.csv tables, one can obtain a wide variety of variables (e.g. surface, city and country of the tournament, nationality of the players etc.), for performance reasons they are not included in the matches.csv table.

The elo.py script uses the Elo algorithm (https://en.wikipedia.org/wiki/Elo_rating_system) to rate the players and calculate win probabilities. One can either set the parameters of the algorithm manually or use the default values which performed well in our tests.

# Coming soon
* We are working on a script prediction.py using a logistic regression to further improve on the win probabilities from the Elo algorithm.
* We have collected weather records from hundreds of stations to be able to calculate the temperature, humidity and wind speed for as many matches as possible.

# ER Diagram

<img src="https://github.com/jakmuell/Tennis/blob/main/ER%20Diagram.jpg" width="700">

