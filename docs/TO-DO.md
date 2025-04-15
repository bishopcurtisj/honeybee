- [x] Remove all unused imports
- [x] Finish Experiment Driver (Testing Phase)
- [x] Finish Controller / Decide what to do about Controller (Testing Phase)
- [x] Write testing script
- [x] Move signal from agents `ndarray` to market
- [x] Refactor
- [x] Replace magic numbers with names representing ID's
- [x] Figure out why pass by reference isn't allowing for the changes to agents by the controller to be reflected in the experiment.

- [x] Set up the ability for agents to purchase information, for now it will be all or nothing.
- [x] Add registration function wrappers to experiment.py
- [x] Update the functions that call systems so that they can store and include extra params.
	- [x] Update learning.py to better match other systems 
- [x] Update the calculations of returns to use a function stored in market so that this change can be made by users easier
- [x] Replace components.id with `components`.`agent_id`
- [ ] Fix genetic algorithm so that it uses the correct indices rather than hard coded ones that may or may not be wrong.
	- [ ] How to handle `num_params`?
- [x] Refactor to remove magic numbers
- [ ] As more utility/objective functions are added, consider refactoring to match demand/spread
- [ ] Fix NN input shape


## Requirements for alpha

- [x] Work on documentation, start making not of what fields each module requires agents to have.
	- [x] Add acceptable values to components
	- [x] Add acceptable pairings of functions to models
- [x] Implement Neural Network agent demand / spread
- [ ] Rewrite agent_constructor.py using new fields
	- [x] Update Demand/Learning Algorithm and demand/info parameters
	- [x] Need to redo how `AgentInfo` handles function parameters.
	- [ ] Add new columns to module documentation
	- [ ] Update documentation to add usage of experiment.py and agent_constructor.py
- [ ] Rewrite test suite and begin debugging 

## High Level
- [ ] Vectorize as many loops as possible
	- [ ] `vmap`: need to learn more
- [x] Refactor to split functions into their own modules, i.e. demand.py, objective.py, spread.py for now it's fine, but as more options are added it'll get messy. It'll also be easier for other people to navigate.
- [ ] Update Jax implementation
- [x] Revisit Experiment.trade() to decide how to induce variance across repetitions without dividend variance. There definitely will be stochasticity, but I want to formalize and understand it better.
	-  Have the Bayesians make a draw for their spread/risk aversion
	-  Neural network should have some variance until it learns enough.
- [x] Move away from Routledge assumptions, since they are too restrictive.
    - [x] Add new learning algorithm's and agent functions
    - [x] Revisit market price calculation
- [x] Revisit fitness calculation to see why I'm getting 0's nan's and inf's
- [x] Revisit Market class typing, dataclass doesn't allow immutable types?

- [x]  Refactor so that functions are split into multiple smaller functions, handle logic to determine which function to call in main function, then direct subsets to each relevant function that can be JIT compiled
- [x] Add ability to add custom demand functions similar to mlflow pyfunc wrappers.
	- [x] ABC class
- [x] Fix Experiment driver, need to make it consistent across functions over whether they are expected to process a single repetition or all of them.
- [x] Fix fitness calculations!!!!!
	- [x] This requires me to make a decision on how to calculate profit. what is the "true" value that they are being judged on? Average price or Last price?
- [x] Implement experiment.update_demands() and finish implementing get_agent_spread
- [x] Refactor so that rather than looping over agents, loop over possible demand functions then 

## Models

- [x] Vectorize get_trades()
- [x] Refactor so that the function to determine informed status follow the same pattern as demand, spread, etc.
- [x] Work on Neural Network
- [x] Refactor neural network to instead learn best trades, then estimate the decision rule that leads to these trades.
- [x] Determine how to approach arguments being passed to each function, some depend on the agent (params), some are experiment wide (trades, crossover/mutation rate)
- [x] Create new entity for trades.
	- [x] Track and append trades from each repetition
- [x] Calculate utility for each trade.
	- [x] Decide how to calculate profit from each trade, what is the "true" value that they are being judged on? Average price or Last price?
- [x] Revisit RL information policy
- [ ] Add ability for Neural Network to set spread
	- [ ] For now just using a dummy approach, will refactor once I decide how I want to approach this.

### Bayesians

- [x] Implement ability for bayesians to be informed
	-  This requires me to decide on whether information should be an incremental function or single purchase.
- [x] Add updating to Bayesian/Thompson Sampling Info Policy
- [x] Need a function to determine behavior for current market (Where should this go?)
	- [x] New spread function
	- [x] New Demand function
- [x] Fix update_priors() so that it allows for different trades for each agent.
- [x] Before Thompson Sampler and BUCB need to implement a vanilla subjective Bayesian
- [x] Need to refactor calculations to change from one-size fits all approach
- [x] Need to implement updating function
- [x] Implement logic for when multiple agents are passed to BayesianDemand
- [ ] Add tau calculation
