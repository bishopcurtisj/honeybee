# Questions

Routledge sets the supply of the assets to a random variable $e_t$. Which is i.i.d 
## Entities

- Traders
- Market?
## Components

**Traders** 
- ID: int -> Just use index of ndarray
- Informed (0 | 1)
- Learning algorithm
	- int that is converted to function using REGISTRY
- Preference relation
	- int that is converted to function using REGISTRY
- Demand Function
	- int that is converted to function using REGISTRY
- Demand
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
	