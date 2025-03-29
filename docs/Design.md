
# Experiment.py

	Acts as the driver for the simulation. 
## Functions

- **learn**
- **trade**
- **calculate_agent_fitness**
	- Need to vectorize this
- **get_agent_spread**


# Agent.py

	 Contains functions used by Agents such as demand and utility

	Current Agent Types:
		0: Trader
			Trader Types:
					0: Linear Demand
					1: Bayesian

### To do 
- Need to add new demand, utility, and objective function options
# Market.py
	Stores values about the market environment that are consistent across all agents.

### To do 
- Need to determine best data class to use, needs to allow for ndarrays, and be mutable if possible

# Calculations.py

	Utility module for performing calculations needed by larger functions.
### To do 

# Learning.py

	Handles the instantiation and registration of learning functions

## Models

### Models.py

Contains abstract class to allow custom learning functions to be written

### Genetic Algorithm
### Thompson Sampling

Agent takes a draw from subjective belief probability distribution for relevant parameters. Then updates their beliefs using Bayes rule. Agent uses binomial distribution to determine whether they should purchase information, a normal distribution for prices, and a geometric for quantity demanded at a given price.

Probability Distribution structures:
- Binomial(p = the agents belief that being informed is the optimal choice)
- Normal(mean, sd)
- Geometric(p = f(price) where f is a function that converts the price into a probability that a unit of the asset should be purchased.)

#### TODO 
- [ ] Set up updating rule
- [ ] Figure out logic for converting price to probability.
### Neural Network

Agent observes the trades that lead to the highest return, then uses a neural network to estimate the decision rule that leads to the optimal trades. 
The neural network is given trade prices and quantities as inputs, and the utility of the trade. Then it learns to predict the utility given the trade price and quantity. During trading this agent uses these predictions to make trades whose predicted utility is above their risk aversion threshold.

### To do 
- Need to add new learning algorithm's
	- BUCB

# Trade.py

	Manages the Limit Order Book that the agents use to trade

### To do 
- For price sorting, what increments should be allowed? 0.01 makes logical sense for real-world comparison, but requires a little extra checks. May save compute time if any searches or iterative methods need to be performed.
- Decide if trade order should be randomized