
Should informed agents instead be able to adjust their behavior for the current period after seeing results of some of the paths.


Downsides to cycling agents through market twice?
- Kind of pointless, since if an agents demand at a price as been met then they wouldn't want to resubmit their orders an purchase more than their demand. Additionally there shouldn't be any offers left that the earlier agents would've taken since they would've been first in the priority queue the entire time. 

Should Bayesian agents use the p-value of a given price as the probability of success in their geometric call? 
- Alternatively they could have a function that allows them to convert price to probability, and perhaps a Thompson sampler could explore different functions.


**Does giving the informed agents more repetitions than the uninformed agents allow prices to be locally constructive and have costly information?**
- Cost function that scales cost as extra repetitions are purchased

How should returns be calculated? Last trade price? Average trade price? Stick with dividends?
- Average makes sense for now, but we will explore both.

Neural Networks should train to predict what quantity to purchase given the price, in order to optimize the reward function (neg utility)

~~**How should neural networks be approached?**
- Training to predict the utility of each trade
	- Spread could be calculated by setting risk aversion to a minimum required utility,
	- Only requires trade information
	- Pretty similar to linear regression
	- Difficult to train on historical data
	- Should the inputs and outputs be flipped? We care more about getting predicted values for the inputs, but this is difficult to do if it's been trained to go the opposite direction. Since returns are a function of both the inputs though predicting specific points for either of them is not very easy (i.e. $2*3 = 3*2$ so should quantity or price be 2)
- Training to predict the price used to calculate returns (mean or last) for each repetition
	- This makes the outputs for training on historical data easier, but the inputs are still difficult
	- How should the spread and quantities be set using this approach.
- Training the network to make trades that optimize mean variance or a similar metric
	- This is probably the best, but significantly more intensive to implement and train
	- Need to think through this one more.
~~Either way it seems like the historical data will need to be labeled or include trade data, perhaps candlestick charts / daily volume would allow for something of this nature to be constructed when making to move to empirical data. For the simulation I'll need to come to a decision here.~~


Revisit Experiment.trade() to decide how to induce variance across repetitions without dividend variance. There definitely will be stochasticity, but I want to formalize and understand it better.