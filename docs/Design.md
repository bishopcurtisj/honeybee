
# Experiment.py

## Purpose
	Acts as the driver for the simulation. 
## Functions

- **learn**
- **trade**
- **calculate_agent_fitness**
	- Need to vectorize this
- **get_agent_spread**


# Agent.py
### Purpose
	 Contains functions used by Agents such as demand and utility

### To do 
- Need to add new demand, utility, and objective function options
# Market.py
	Stores values about the market environment that are consistent across all agents.

### Purpose

### To do 
- Need to determine best data class to use, needs to allow for ndarrays, and be mutable if possible

# Calculations.py
### Purpose
	Utility module for performing calculations needed by larger functions.
### To do 

# Learning.py
### Purpose
	Contains the implementations of possible learning functions for agents

### Options
 - Genetic Algorithm
 - Thompson Sampling
	 - Takes a draw from binomial to determine informed status
	 - Takes draw from Demand functions
	 - Takes draw from prior distribution over demand function parameters
	 - Maybe I should represent the competing draws as different agents that have a flag indicating they are the same agent. This poses issues though such as interaction across phantom agents. With multiple Thompson samplers there would be a lot of noise since each agents potential paths would affect the decisions of every other agent. In order to do this properly we'd likely need to simulate every permutation which get's very large very quickly. Alternatively perhaps the law of large numbers will allow us to converge to the true performance of decisions by simply including that Thompson agent version in sufficient simulations. Maybe each repetition the agent is swapped between the options 

### To do 
- Need to add new learning algorithm's
	- Thompson Sampling
	- Neural Networks
	- BUCB

# Trade.py
### Purpose
	Manages the Limit Order Book that the agents use to trade

### To do 
- For price sorting, what increments should be allowed? 0.01 makes logical sense for real-world comparison, but requires a little extra checks. May save compute time if any searches or iterative methods need to be performed.
- Decide if trade order should be randomized