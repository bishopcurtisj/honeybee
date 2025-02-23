# Questions

Routledge sets the supply of the assets to a random variable $e_t$. Which is i.i.d 



### Limit order book vs Intersecting demand/supply

Both?
	An agent calculates their demand for a good based on it's price. If their demand is negative then they become a supplier. The bid-ask spread allows them to express uncertainty, with a continuous demand function the spread would be zero. Can create a spread by allowing the agent to set a confidence/risk interval to set the price. Informed traders will have narrower spreads due to higher confidence in their predictions. 

	 Demand/supply is a function of their valuation and the markets valuation. Bid and ask can be derived by finding the intercept and adding bands around the intercept. Similar to momentum/mean reverting trading strategies where the signal isn't triggered until the metric exceeds a certain z-score. 

	Agents calculate their bid ask spreads and then submit orders with quantity willing to transact and the price. If a linear and continous demand function is used then an example might be that they are unwilling to trade when their demand is +-X from zero where X would represent the agents confidence in their demand function.

### Limit Order Book Design

Should I use a queue, a matching, or an auction
	A queue seems easiest to implement and leads to less follow up questions such as how to handle spread or what price it is executed at.
## Entities

- Agents
- Market?
	- Unique object, named tuple with values such as price.
- Controller
	- Handles the sending of relevant values from the agents ndarray to  each function
## Components

**Traders** 
- ID: int -> Just use index of ndarray?
- Informed (0 | 1)
- Learning algorithm
	- int that is converted to function using REGISTRY
- Utility Function
	- int that is converted to function using REGISTRY
- Demand Function
	- int that is converted to function using REGISTRY
- Demand
	- Negative demand = supply
- Objective Function
	- int that is converted to function using REGISTRY
- Fitness
- Function parameter values
- Signal
- Previous Period Return
- Aggregate Return
- Cost of signal

**Market**
- Price
- Demands

## Systems

Initializer
	Initializes the agents and experiment given the details in agents.csv and components.json
	

Agent learning algorithm
	

Trading mechanism
	Market uses iterative methods to identify the price that balances demands
	Routledge uses randomly set supply rather than coordinating bid asks