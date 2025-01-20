
## Entities

- Traders
- Market?
## Components

**Traders** 
- ID: int
- Informed (0 | 1)
- Learning algorithm
	- int that is converted to function using REGISTRY
- Preference relation
	- int that is converted to function using REGISTRY
- Demand Function
	- int that is converted to function using REGISTRY
- Objective Function
	- int that is converted to function using REGISTRY
- Function parameter values
- Signal
- Previous Period Return
- Aggregate Return

**Market**
- Price
- Demands

## Systems

Initializer
	Initializes the agents and experiment given the details in agents.csv and components.json

Agent learning algorithm
	

Trading mechanism
	Market uses iterative methods to identify the price that balances demands
	