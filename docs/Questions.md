Memory vs performance. flag?

Do Thompson agents need to be paired with a duplicate that does the opposite of their levers in order to determine the best lever to pull?

Agents see simulations of trading and then have one round where they trade 
True rounds are weighted heavier than simulations

Should Neural Networks and GA share the informed approach with Thompson?

**Does giving the informed agents more repetitions than the uninformed agents allow prices to be locally constructive and have costly information?**
- Cost function that scales cost as extra repetitions are purchased

How should returns be calculated? Last trade price? Average trade price? Stick with dividends?

How should neural networks be approached?
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


Revisit Experiment.trade() to decide how to induce variance across repetitions without dividend variance. There definitely will be stochasticity, but I want to formalize and understand it better.