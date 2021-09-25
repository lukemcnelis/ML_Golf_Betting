# ML_Golf_Betting
Fourth Brain Capstone Project

Golf is a very difficult sport to make predictions in due to a number of factors:

●	Large number of players: Field sizes vary from 30-156 players per event

●	Weather: Sun, wind and rain all play a big role in scoring.  Tournaments are played over 4 days with the weather constantly changing

●	Variety of course setups: Links / parkland / length / grass types / rough length / green speed etc.

●	Variety of locations: Tournaments are played all over the world with different climates and conditions affecting players differently

●	Player form: All professional players go through several peaks and troughs in their career.  These are very difficult to predict and a professional golfer can go from hero-to-zero and vice versa for no apparent reason

●	Luck: The bounce of the ball can play a large roll in outcomes

As such, success rates of predictions are low relative to predictions in other sports, e.g., in Football, Tennis, Horse Racing etc.  This is reflected by the high odds that one can get from bookmakers - favourites typically come in around 6-12/1, midrange players around 20-80/1 and the rest can vary from 100-1000/1.

It is also a data rich sport, with statistical analysis now impacting every aspect of the game from player improvement, coaching techniques and club development, to course setup and fan engagement.  

Off the back of this (as with all modern sports), there are now a wide array of sports betting opportunities around professional events, including – outright betting, top10 finishes, head-to-head matchups and fantasy sports team building – to name but a few.  

In my opinion, all of the above makes golf a target rich environment for AI & ML opportunities.  For the past five years I have been systematically collecting data from the US PGA & European professional golf tours.  I now have a bespoke dataset that includes:

a)	Data from over 600 PGA & European tour events dating back to 2014

b)	Results for over 3,400 professional golfers across these events

o	a) & b) have been combined to create a dataset with 59,000+ records (m) and 323 custom features (n) summarizing individual tournament performances and results

c)	Records for 9,700+ head-to-head matchups (going back to 2017) with odds data from betting exchanges  

For the individual learning project, I have been applying the concepts and techniques (from weeks 0-4) to a specific subset of this data as outlined below:

PGA Tour Round 2 (R2), 3-ball head-to-head winner prediction:

●	m (train & test) = 1,415 | # of unique R2, 3-ball head-to-head matchups from the start of 2017 to the end of the 2020 PGA tour season

o	m_p = 4,245 (1,415 x 3) | # of unique player records.  Each ‘m’ (3-ball) has 3 players so ‘m_p’ represents the total # of unique player records in the dataset

●	m (hold out) = 291 | # of unique R2, 3-ball head-to-head matchups for the 2021 PGA tour season

o	m_p = 873

●	n = 312 | # of features per player record used for training and prediction purposes.  These can be classified into 4 groups:

o	Historical data averaging a player’s results over the prior 2 years on similar courses, settings, types of tournaments, etc.

o	Previous tournament data summarizing a player’s performance (relative to the rest of the field) going back over the previous 12 tournaments that they played

o	Round 1 performance data (from the previous day’s play)

o	Round 2 group betting odds

To date, I have trained and evaluated 3,409 different model variants with the aim of identifying the best performing options in terms or RoR (Return on Risk = accumulated profit / accumulated wagers).  Below is a snapshot of the top performing ones:
 

These preliminary findings are encouraging.  My goal is to research and develop a suite of models capable of delivering consistent returns across all the head-to-head betting opportunities for the PGA and European tours, this equates roughly to: 

●	~2,500 head-to-head matchups per year 

●	Spread over ~65 tournaments 

●	Occurring over ~35 weeks

If successful, the project will deliver a revenue generating ML platform capable of producing consistent returns (tax free!) on wagers.
The graph below illustrates the theoretical return on a bankroll of $200 (20 x $10 bets per day when a PGA/EUR event is on) over the course of the season.  It is based on an average RoR of 5% across the ~2,560 available head-to-head matchups

●	Note: The best model from my individual project so far produced an average RoR of 5.49% across all PGA R2 3ball matchups
  
●	Initial bankroll = $200

●	Average 5% RoR over season = $1,280 profit, representing a 640% return on bankroll

●	The above calculation is based on wagering $10 on each of the ~2,560 available head-to-head matchups in a year.  On average there are 20 matchups per day when there is a tournament on, which is how the $200 bankroll figure is calculated.

Inspiration for this project was taken from the story below about Bill Benter who developed an algorithm that couldn’t lose at horse racing

https://www.bloomberg.com/news/features/2018-05-03/the-gambler-who-cracked-the-horse-racing-code

https://www.bloomberg.com/news/videos/2020-01-09/the-man-who-beat-horse-racing-and-made-close-to-a-billion-dollars-video
