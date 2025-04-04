- [x] Remove all unused imports
- [x] Finish Experiment Driver (Testing Phase)
- [x] Finish Controller / Decide what to do about Controller (Testing Phase)
- [x] Write testing script
- [x] Move signal from agents ndarray to market
- [x] Refactor
- [x] Replace magic numbers with names representing ID's
- [x] Figure out why pass by regerence isn't allowing for the changes to agents by the controller to be reflected in the experiment.
- [ ] Rewrite agent_constructor.py 
## High Level
- [x] Move away from Routledge assumptions, since they are too restrictive.
    - [x] Add new learning algorithm's and agent functions
    - [x] Revisit market price calculation
- [x] Revisit fitness calculation to see why I'm getting 0's nan's and inf's
- [x] Revisit Market class typing, dataclass doesn't allow immutable types?
- [ ] Vectorize as many loops as possible
- [ ] Update Jax implementation
- [x]  Refactor so that functions are split into multiple smaller functions, handle logic to determine which function to call in main function, then direct subsets to each relevant function that can be JIT compiled
- [x] Add ability to add custom demand functions similar to mlflow pyfunc wrappers.
	- [x] ABC class
- [x] Fix Experiment driver, need to make it consistent across functions over whether they are expected to process a single repetition or all of them.
- [x] Fix fitness calculations!!!!!
	- [x] This requires me to make a decision on how to calculate profit. what is the "true" value that they are being judged on? Average price or Last price?
- [x] Implement experiment.update_demands() and finish implementing get_agent_spread
- [x] Refactor so that rather than looping over agents, loop over possible demand functions then 
- [ ] Revisit Experiment.trade() to decide how to induce variance across repetitions without dividend variance. There definitely will be stochasticity, but I want to formalize and understand it better.
## Models

- [x] Refactor neural network to instead learn best trades, then estimate the decision rule that leads to these trades.

- [x] Determine how to approach arguments being passed to each function, some depend on the agent (params), some are experiment wide (trades, crossover/mutation rate)
- [x] Create new entity for trades.
	- [x] Track and append trades from each repetition
- [ ] Vectorize get_trades()
- [x] Calculate utility for each trade.
	- [ ] Decide how to calculate profit from each trade, what is the "true" value that they are being judged on? Average price or Last price?

### Bayesians
- [x] Before Thompson Sampler and BUCB need to implement a vanilla subjective Bayesian

- [ ] Implement ability for bayesians to be informed
	-  This requires me to decide on whether information should be an incremental function or single purchase.
- [x] Need to refactor calculations to change from one-size fits all approach
- [ ] Need a function to determine behavior for current market (Where should this go?)
	- [x] New spread function
	- [x] New Demand function
- [x] Need to implement updating function
- [x] Implement logic for when multiple agents are passed to BayesianDemand
