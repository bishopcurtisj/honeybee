# Questions

Routledge sets the supply of the assets to a random variable $e_t$. Which is i.i.d 

Should we use a controller to send the array slices to functions, or build the parsing into the functions.

How flexible should I allow the structure of agents to be?



Limit order book vs Intersecting demand/supply

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