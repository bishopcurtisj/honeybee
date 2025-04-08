This package is designed to provide a modular and extensible framework for Agent Based Simulations in Financial Markets. This document outlines the structure of the package, current features, and required agent fields for each tuning option.

# Structure
`src/`  
├── [`experiment.py`](#experiment)  
├── [`globals.py`](#globals)  
├── [`entities`](#entities)  
│   ├── [`agents.py`](#agents)  
│   ├──[`market.py`](#market)  
│   └── [`trades.py`](#trades)  
└── [`systems`](#learning)  
    ├── [`learning.py`](#learning)  
    ├── [`trade.py`](#trade)  
    ├── [`agent_functions`](#agent_functions)  
    │   ├── [`demand.py`](#demand)  
    │   ├── [`objective.py`](#objective)  
    │   ├── [`spread.py`](#spread)  
    │   └── [`utility.py`](#utility)  
    └── [`models`](#models)  
        ├── [`model.py`](#model)  
        ├── [`bayesian.py`](#bayesian)  
        ├── [`genetic_algorithm.py`](#genetic_algorithm)  
        ├── [`neural_network.py`](#neural_network)  
        ├── [`loss.py`](#loss_functions)  
        └── [`information_policy.py`](#information_policy)  

# `Experiment`

## Purpose

The `experiment` module contains the main driver class `Experiment` which coordinates the entire lifecycle of a simulation run. This includes initializing agent and market state, executing trades across multiple generations and repetitions, applying learning algorithms, updating agent parameters, and saving results. It is designed to be the top-level orchestrator of an experimental configuration using the ACE (Agent-based Computational Economics) framework.

The `Experiment` class encapsulates all critical steps of the simulation loop:
1. Data loading and setup from configuration files.
2. Trading interactions using a priority queue order book.
3. Agent learning using registered models.
4. Fitness evaluation using custom-defined objectives.
5. Optionally saving experiment results to disk.

## Custom Option Requirements

This module includes support for registering custom strategies and behavior systems used by agents. To plug in a custom system, users must define a class/function following the appropriate interface and use the corresponding registration method.

### Customizable Systems:

- **Demand Function** (`register_demand_function`)  
  Must subclass `Demand` and implement a callable interface returning quantity demanded.
  
- **Objective Function** (`register_objective_function`)  
  Must subclass `Objective` and implement an evaluation method for expected utility or similar.
  
- **Spread Function** (`register_spread_function`)  
  Must subclass `Spread`. Used to define bid-ask logic based on confidence or signals.
  
- **Utility Function** (`register_utility_function`)  
  Must subclass `Utility`. Converts outcomes into scalar utilities.
  
- **Loss Function** (`register_loss_function`)  
  Must subclass `AgentLoss`. Used by learning systems to evaluate performance.
  
- **Information Policy** (`register_information_policy`)  
  Must subclass `InformationDecisionPolicy`. Governs how agents decide whether to purchase signals.

- Learning Algorithm (`register_learning_algorithms`)
  Must subclass Model. Governs how agents learn across generations.

## Function Signatures

```python
class Experiment:
    def __init__(self, market: Market, agents_file_path: str, config_file_path: str)

    def run(self, generations: int = 20, repetitions: int = 100) -> jnp.ndarray

    def save(self)

    def learn(self)

    def trade(self) -> jnp.ndarray

    def calculate_agent_fitness(self, trades: jnp.ndarray)

    def get_agent_spread(self)

    def register_demand_function(self, demand_functions: Union[List[Demand], Demand])

    def register_objective_function(self, objective_functions: Union[List[Objective], Objective])

    def register_spread_function(self, spread_functions: Union[List[Spread], Spread])

    def register_utility_function(self, utility_functions: Union[List[Utility], Utility])

    def register_loss_function(self, losses: Union[List[AgentLoss], AgentLoss])

    def register_information_policy(self, info_policies: Union[List[InformationDecisionPolicy], InformationDecisionPolicy])
    
    def register_learning_algorithms(self, models: Union[List[Model], Model]):
```

## Used Components

The `Experiment` class uses a wide variety of components attached to agent entities. These include:

- `fitness`: _Tracks an agent’s performance over time._
- `informed`: _Binary flag indicating whether the agent has access to noisy price signals._
- `signal`: _The agent’s perceived asset value if informed._    
- `bid`, `ask`: _Price limits for potential trades._    
- `bid_quantity`, `ask_quantity`: _Quantity offered at bid/ask prices._
- `demand`, `demand_function`: _Stores and calls the demand calculation logic._
- `objective_function`: _Determines how agent fitness is evaluated._
- `utility_function`: _Converts returns into scalar utilities for comparison._
- `spread_function`: _Used to determine the bid-ask spread based on confidence and other features._
- `risk_aversion`: _Used in utility or fitness calculations and for setting Bayesian spread._
- `learning_params`: _The parameters passed to agent models during learning._
- `agent_type`: _Identifies types of agents (e.g., active traders, passive)._
- `id`: _Unique agent identifier._
# `Globals`
## Purpose

The `globals` module defines two key classes, `Globals` and `Config`, which act as centralized repositories for simulation-wide state and configuration parameters. These classes are instantiated once (`globals` and `config`, respectively) and imported across the codebase to provide shared access to market conditions, agent data, experiment settings, and runtime metadata.

This module eliminates the need to pass large context objects between function calls, simplifying the integration and modular design of the framework.

## Function Signatures

```python
class Globals:
    agents: jnp.ndarray
    components: AgentInfo
    market: object
    trades: jnp.ndarray
    informed: bool
    generation: int
```
- **agents**: A 2D `jnp.ndarray` representing the current state of all agents in the simulation.
- **components**: An `AgentInfo` object that maps agent feature names to column indices.
- **market**: An instance of the current `Market` class being simulated.
- **trades**: Records results of each period’s trading activity (e.g., quantities, prices).
- **informed**: Boolean flag indicating whether the simulation contains informed agents.
- **generation**: Current generation index in the simulation loop.
```python
class Config:
    uninformed_base_ratio: float
    mutation_rate: float
    crossover_rate: float
    generations: int
    repetitions: int
    GAMMA_CONSTANTS: List
    max_price: float
    memory_optimization: bool = True
    save_models: bool = True

    def from_json(self, json_path: str)
```
- **uninformed_base_ratio**: Base probability that an agent is uninformed.
- **mutation_rate** / **crossover_rate**: Genetic algorithm tuning parameters.
- **generations** / **repetitions**: Core experiment loop parameters.
- **GAMMA_CONSTANTS**: A list of constants used in agent-specific calculations.
- **max_price**: Upper bound for price space in simulations.
- **memory_optimization**: Toggles tradeoffs in RAM vs. compute usage.
- **save_models**: If `True`, learned models will be saved during simulation.
### Method
- `from_json(json_path: str)`: Loads configuration values from a JSON file and sets the relevant fields on the `Config` object.

# Entities
## `Agents` 

### Purpose

The `agent` module defines the `AgentInfo` class, which serves as a centralized, dynamically generated registry of column indices for agent-related data stored in `globals.agents`. This system allows all modules to reference agent attributes using human-readable keys (like `fitness`, `bid`, `signal`, etc.), while enabling efficient access to underlying `jnp.ndarray` rows using column indices.

This setup is a critical element of the Entity Component System (ECS) architecture, enabling both flexibility and performance.

### Class Signature

```python
class AgentInfo:
    def __init__(self, columns: List[str])
    def __getitem__(self, key: Union[str, int])
    def add(self, name: str, index: int)
    def keys(self) -> List[str]
    def values(self) -> List[int]
    def items(self) -> Iterator[Tuple[str, int]]
    def __iter__(self)
```
#### Special Attributes

- `demand_fx_params`: List of column indices for all fields prefixed with `dfx_`. Used by demand and learning models.
- `learning_params`: List of column indices for all fields prefixed with `la_`. Used by learning algorithms that need to be initialized or each agent.
- `info_params`: List of column indices for all fields prefixed with `info_`. Used by information decision policies that need to be initialized or each agent.
#### Core Features

- Dict-style access: `components["fitness"]` or `components.fitness`
- Dynamic growth via `add(name, index)`
- Used globally as `globals.components` throughout the framework
---
### Components

| Field                | Description                                                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `fitness`            | Agent fitness score, updated by learning algorithms                                                                           |
| `informed`           | Binary flag (0 or 1) indicating whether an agent receives a signal                                                            |
| `signal`             | The noisy valuation signal observed by informed agents                                                                        |
| `bid`                | Price at which the agent is willing to buy                                                                                    |
| `ask`                | Price at which the agent is willing to sell                                                                                   |
| `bid_quantity`       | Quantity offered at the bid price                                                                                             |
| `ask_quantity`       | Quantity offered at the ask price                                                                                             |
| `demand`             | Raw demand value computed from the agent’s demand function                                                                    |
| `demand_function`    | Index into the `DEMAND_REGISTRY` used by this agent                                                                           |
| `demand_fx_params`   | Auto-extracted list of parameters used in demand estimation                                                                   |
| `objective_function` | Index into `OBJECTIVE_REGISTRY` to determine fitness calculation strategy                                                     |
| `utility_function`   | Index into `UTILITY_REGISTRY` used to evaluate expected utility                                                               |
| `risk_aversion`      | Used in utility and fitness functions, esp. CARA models                                                                       |
| `learning_algorithm` | Index into model registry, determines agent’s learning strategy                                                               |
| `learning_params`    | Hyperparameters used by learning algorithms                                                                                   |
| `spread_function`    | Index into `SPREAD_REGISTRY`, used to determine bid-ask spread logic                                                          |
| `confidence`         | Determines spread width for Agents, this should be a dollar amount for non-Bayesian agents, and an alpha level for Bayesians. |
| `loss`               | Index into `LOSS_REGISTRY`, used in neural net agents                                                                         |
| `loss_params`        | Hyperparameters passed to the loss function constructor                                                                       |
| `information_policy` | Index into `INFORMATION_POLICY_REGISTRY` for info acquisition logic                                                           |
| `info_params`        | Hyperparameters passed to InformationDecisionPolicy constructors                                                              |
| `agent_type`         | Used to filter agents (e.g., only active traders) during trading routines                                                     |
| `agent_id`           | Unique identifier for each agent, often used as a key in dictionaries                                                         |
**Note:** These components are used in at least one of the implemented functions, not all will be necessary depending on the type of simulation you are running.
## `Market`

### Purpose

The `market` module defines the abstract `Market` interface and two specific implementations: `GSOrderMarket` and `RoutledgeMarket`. These classes encapsulate market-level dynamics, including supply generation, dividend processes, and signal-noise construction across generations and repetitions.

Each market maintains its own state (prices, signals, dividends, etc.) and resets this state at the beginning of each new period via the `new_period()` method. The `GSOrderMarket` also manages a full order book using price-time priority, making it the standard environment for trading simulations in this framework.

### Function Signatures

```python
class Market(ABC):
    dividends: np.ndarray
    price: float = 0.0
    last_period_price: float = 0.0
    repetitions: int
    generations: int
    supply: np.ndarray
    demand_at_p0: float
    last_price: float
    mean_price: float

    @abstractmethod
    def new_period(self)
```
---
#### `GSOrderMarket`
```python
class GSOrderMarket(Market):
    order_book: OrderBook
    cost_of_info: float
    signal: np.ndarray
    noise: np.ndarray
    beta0: float
    beta1: float
    dividends: np.ndarray
    supply: np.ndarray
    last_price: jnp.ndarray
    mean_price: jnp.ndarray

    def new_period(self)
```
- Simulates a Grossman-Stiglitz-style market.
- Updates signal and dividend processes each generation.
- Includes a full `OrderBook` for execution-level trade handling.
---
#### `RoutledgeMarket`
```python
class RoutledgeMarket(Market):
    cost_of_info: float
    signal: np.ndarray
    noise: np.ndarray
    beta0: float
    beta1: float
    dividends: np.ndarray
    supply: np.ndarray

    def new_period(self)
```
- Designed to replicate Routledge (2001) dynamics.
- Similar statistical process to `GSOrderMarket`, but lacks an order book.
---

## Trades
Unsure if this will stay, it isn't currently being used

# `Learning`

## Purpose

The `learning` module acts as the central controller for agent learning behavior in the simulation. It defines the `ModelController` class, which manages initialization, registration, and execution of learning models applied to subsets of agents based on their `learning_algorithm` ID. This controller abstracts the routing logic between different learning strategies such as neural networks, genetic algorithms, and Bayesian agents.

It ensures that:
- Each agent uses the correct learning system based on internal identifiers.
- New learning algorithms can be added via a registration interface.
- Model updates are efficiently applied each generation during the `learn` phase.

## Custom Option Requirements

To add a custom learning algorithm, subclass `Model` and register it using `ModelController.register_models(...)`. Each model must define:
- A `label` (string) used to identify it in the registry.
- A callable `__call__` method that takes a subset of agents and returns updated agent states.
- Optional `args` if the model needs runtime arguments.
- If your learning algorithm is unable to use a `@staticmethod __call__()`, you must instantiate it and then register the instantiated object. If each agent needs it's own instance, utilize the neural network approach as a guide to do so.

```python
# Example custom model registration
from systems.models.model import Model

class CustomLearner(Model):
    label = "custom"

    def __call__(self, agents, *args, **kwargs):
        ...

model_controller.register_models(CustomLearner)
```
---
## Function Signatures

```python
class ModelController:
    def init_models(self)

    def register_models(self, models: Union[List[Model], Model])

    def learn(self) -> jnp.ndarray

```
---
### Method Descriptions

- `init_models()`:  
    Initializes built-in models (neural network, genetic algorithm, Bayesian). Identifies which agents use each model via the `learning_algorithm` component.
- `register_models(models)`:  
    Adds new learning models to the registry. Accepts either a single model or a list. Each model must be a subclass of `Model` and define a unique `label` and `id`.
- `learn() -> jnp.ndarray`:  
    Calls the `__call__` method of each registered learning algorithm on the appropriate agent subset. The updated agents array is stored back into `globals.agents`.
## Registry

The registry accesses the proper algorithm using the agents `learning_algorithm` component, the options are currently:

| Key | Algorithm      |
| --- | -------------- |
| 1   | Genetic        |
| 2   | Bayesian       |
| 3   | Neural Network |
Newly registered custom options are appended in order, i.e. the first custom model will have the key 4.
## Used Components
- `learning_algorithm` the registry id for the learning algorithm the agent uses.
# `Trade`

## Purpose

The `trade` module implements a priority queue-based order book system for handling market transactions between agents. It defines two primary classes:

- `Order`: A simple structure representing a bid or ask with price, quantity, and trader identity.
- `OrderBook`: A double-sided priority queue that matches buy and sell orders using price-time priority.

This module simulates market microstructure dynamics, where agents submit buy/sell orders which are then matched against the book if possible. It supports partial matches, stores unmatched orders, and tracks all trades executed during a given trading round.

## Function Signatures

```python
class Order:
    def __init__(self, trader_id: str, price: float, quantity: float)

    def __repr__(self) -> str
```
- `trader_id`: The agent submitting the order.
- `price`: The bid or ask price.
- `quantity`: Quantity of the asset. Positive for buy, negative for sell.
---
```python
class OrderBook:
    def __init__(self)

    def reset(self)

    def add_order(self, trader_id: str, price: float, quantity: float)

    def match_orders(self, order: Order) -> Order

    def get_agent_trades(self) -> dict[str, jnp.ndarray]

    def get_trades(self) -> jnp.ndarray
```
---
### Method Descriptions

- `reset()`: Clears all buy/sell queues and trade history. Called at the end of each trading repetition.
- `add_order(...)`: Submits an order to the order book and attempts to match it. Unmatched remainder is added to the book.
- `match_orders(...)`: Internal method that implements trade matching logic, prioritizing best price first and earliest arrival second.
- `get_agent_trades()`: Returns a dictionary mapping agent IDs to their trade histories (quantities and prices).
- `get_trades()`: Returns a full array of all trades with an additional column showing price deviation from the mean, directionally signed by buyer/seller role.
## Used Components
- `agent_id`: _Unique agent identifier._
- `bid`, `ask`: _Price limits for potential trades._    
- `bid_quantity`, `ask_quantity`: _Quantity offered at bid/ask prices._
## Trade Output Format

The result of `get_trades()` is a matrix of shape `[n_trades, 3]` where each row represents:
- `quantity`: Positive for buy-side, negative for sell-side.
- `price`: Transaction price.
- `return`: Signed deviation from the mean trade price, weighted by transaction direction.

# Agent_Functions
## Demand

### Purpose

The `demand` module defines how agents compute the quantity of an asset they are willing to buy or sell at a given price. It provides a flexible interface for implementing and registering different demand functions, allowing heterogeneous agent behaviors. Demand functions can reflect linear utility, probabilistic reasoning, or other forms of decision-making.

Each demand function is implemented as a subclass of the abstract base class `Demand`, which requires a static `__call__` method that returns a scalar quantity.

### Custom Option Requirements

To define a custom demand function:
1. Subclass `Demand`.
2. Assign a unique `name`.
3. Implement a static `__call__` method that returns a float quantity based on inputs such as price, agent beliefs, or signals.
4. Register the custom function using `register_demand_function(...)`.

#### Example:

```python
class MyCustomDemand(Demand):
    name = "MyCustom"

    @staticmethod
    def __call__(price, params, signal):
        return some_quantity

register_demand_function(MyCustomDemand)
```
### Function Signatures

```python
class Demand(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float
```
---
```python
class GS_linear(Demand):
    name = "GS_linear"

    @staticmethod
    def __call__(
        price: float, coeffs: jnp.ndarray, signal, scaling_factor: float = None
    ) -> float
```
- Linear demand function used in Grossman-Stiglitz-type settings.
- Computes demand as:  
    `scaling_factor * (intercept + slope * signal - price)`
---
```python
class BayesianDemand(Demand):
    name = "BayesianDemand"

    @staticmethod
    def __call__(price: jnp.ndarray, bid: bool, params: jnp.ndarray, key=None) -> float
```
- Models agent demand as a geometric draw based on the probability of price under the agent’s subjective normal distribution.
- Uses JAX's `geometric` RNG function for vectorized sampling.
- Parameters:    
    - `price`: Current market price
    - `bid`: `True` if buying, `False` if selling
    - `params`: Array of shape `[mean, std]` representing the agent’s belief about price
    - `key`: Optional PRNG key for JAX
---
### Registry

The registry maps agent `demand_function` component values to callable demand logic. The currently available options are:

| Key | Demand Function  |
| --- | ---------------- |
| 1   | GS_linear        |
| 2   | BayesianDemand   |

Newly registered custom options are appended in order. For example, the next custom function registered will receive the key `3`.
### Used Components

This module expects to be used with the following agent components:
- `demand_function`: An identifier that maps an agent to a demand function in `DEMAND_REGISTRY`
- `demand_fx_params`: Coefficients or parameters passed to the selected demand function.
- `signal`: (Optional) An informed agent's signal, used in linear demand or Bayesian reasoning.
- `price`: The price of the asset, typically passed in from the market context.
## Objective

### Purpose

The `objective` module defines how agent fitness is evaluated during the simulation. It provides an abstract `Objective` class and a registration system to support flexible evaluation strategies. Objective functions are responsible for computing a scalar fitness score based on an agent’s realized utility and risk preferences.

This module also includes a vectorized `calculate_fitness` function that applies the appropriate objective function to each agent during the learning phase.

### Custom Option Requirements

To define a custom objective function:
1. Subclass `Objective`.
2. Assign a unique `name`.
3. Implement a static `__call__` method that returns a fitness score from utility and risk-aversion values.
4. Register it with `register_objective_function(...)`.

#### Example:

```python    
class MyObjective(Objective):
    name = "MyObjective"

    @staticmethod
    def __call__(utilities: jnp.ndarray, risk_aversion: float) -> float:
        return jnp.mean(utilities) - risk_aversion * jnp.max(utilities)

register_objective_function(MyObjective)
```
### Function Signatures

```python
class Objective(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float
```
---
```python
class Mean_variance(Objective):
    name = "Mean_variance"

    @staticmethod
    def __call__(utilities: jnp.ndarray, risk_aversion: float) -> float
```
- Computes a mean-variance utility tradeoff:  
	`mean(utilities) - 0.5 * risk_aversion * var(utilities)`
---
```python
def calculate_fitness(
    agents: jnp.ndarray, trades: jnp.ndarray, risk_aversion: jnp.ndarray
) -> jnp.ndarray
```
- Applies each agent’s chosen objective function to compute and store fitness.
- Relies on utility values produced via `calculate_utility(...)`.
---
```python
def calculate_returns(agents: jnp.ndarray, trades: jnp.ndarray) -> jnp.ndarray
```
- Computes net asset returns by subtracting total cost from value (including signal cost if informed).
- Currently uses a mean transaction price model.
---
### Registry

The registry maps the `objective_function` component to callable objective functions. The currently available options are:

|Key|Objective|
|---|---|
|1|Mean_variance|

New custom objectives are appended in order. For example, the next registered objective function will use key `2`.
### Used Components

- `fitness`: Target column for storing each agent’s computed score.
- `objective_function`: Selects the appropriate strategy from `OBJECTIVE_REGISTRY`.
- `utility_function`: Required by `calculate_utility(...)`, indirectly used.
- `informed`, `signal`, `demand`, `demand_function`: Passed through agent rows for consistency and use in utility/return calculations.
- `demand_function_params`: Parameters passed into demand estimation and utility computation.
- `risk_aversion`: Used directly in most objective functions.
## Spread

### Purpose

The `spread` module defines how agents determine their bid-ask spread based on confidence, signals, and demand. It provides a standardized interface for implementing spread strategies through the abstract `Spread` class. Registered spread functions are used by the experiment engine to populate agent order details (price and quantity) in each repetition.

The two default spread functions represent:
- Linear demand agents computing price inversions from demand equations.
- Bayesian agents setting spread bounds using confidence intervals from a subjective distribution.

### Custom Option Requirements

To define a custom spread function:
1. Subclass `Spread`.
2. Assign a unique `name`.
3. Implement a `__call__` method that modifies agent state in-place.
4. Register with `register_spread_function(...)`.

#### Example

```python   
class MySpread(Spread):
    name = "MySpread"

    @staticmethod
    def __call__(agents: jnp.ndarray) -> jnp.ndarray:
        # Update bid/ask using some logic
        return agents

register_spread_function(MySpread)
```
### Function Signatures

```python
class Spread(ABC):
    name: str

    @abstractmethod
    def __call__(*args, **kwargs) -> float
```
---
```python
class LinearDemandSpread(Spread):
    @staticmethod
    def __call__(agents: jnp.ndarray) -> jnp.ndarray
```
- Computes a bid-ask midpoint using a root-finding method to invert demand to price.
- Applies different logic for informed and uninformed agents.
- Quantities are calculated from the chosen bid/ask prices using the agent's demand function.
---
```python
class BayesianSpread(Spread):
    @staticmethod
    def __call__(agents: jnp.ndarray) -> jnp.ndarray
```
- Uses normal distribution quantiles to set bid/ask prices based on agent confidence levels.
- Quantities are generated by calling the registered demand function at each bound.
### Registry

The registry maps the `spread_function` component in each agent to a spread strategy. The current built-in options are:

|Key|Spread Function|
|---|---|
|1|LinearDemandSpread|
|2|BayesianSpread|

New custom functions are assigned incremental keys in the order they are registered.
### Used Components

- `informed`: Indicates whether to use signal-based (informed) or prior-free (uninformed) logic.
- `signal`: Used as a center point for price confidence intervals or demand inversions.
- `bid`, `ask`: Prices generated by the spread function.
- `bid_quantity`, `ask_quantity`: Computed demand at each price bound.
- `confidence`: Determines spread width for agents.
- `demand_function`: Required to compute implied quantities.
- `demand_function_params`: Parameters passed into the registered demand function.
## Utility

### Purpose

The `utility` module defines how agents transform financial returns into scalar utilities, which are then used by learning algorithms and objective functions. It provides a flexible interface through the `Utility` abstract class and supports registering custom utility functions that capture different risk preferences.

The default implementation uses **Constant Absolute Risk Aversion (CARA)** preferences, but the system can support any transform mapping returns and risk parameters to scalar utility.

### Custom Option Requirements

To create a custom utility function:
1. Subclass `Utility`.
2. Assign a unique `name`.
3. Implement a static `__call__` method that takes returns and risk preferences as inputs.
4. Register the class using `register_utility_function(...)`.

#### Example

```python
class Utility(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float
    
class MyUtility(Utility):
    name = "MyUtility"

    @staticmethod
    def __call__(returns: jnp.ndarray, risk_aversion: jnp.ndarray) -> jnp.ndarray:
        return returns ** (1 - risk_aversion)

register_utility_function(MyUtility)
```
### Function Signatures

```python
class Const_abs_risk_aversion(Utility):
    name = "Const_abs_risk_aversion"

    @staticmethod
    def __call__(returns: jnp.ndarray, risk_aversion: jnp.ndarray) -> jnp.ndarray
```
- Implements the CARA utility function:  
	`U(x) = -exp(-risk_aversion * x)`
---
```python
def calculate_utility(
    agents: jnp.ndarray, returns: jnp.ndarray, risk_aversion: jnp.ndarray
) -> jnp.ndarray
```
- Computes utility values for each agent, using the appropriate registered utility function.
- Returns a matrix of shape `(n_agents, n_repetitions)`.
- ---
### Registry

The registry maps the `utility_function` component in each agent to a callable utility strategy. Current options include:

|Key|Utility Function|
|---|---|
|1|Const_abs_risk_aversion|

Custom utility functions are appended sequentially. The next registered function will have the key `2`.
### Used Components

- `utility_function`: Integer used to select from `UTILITY_REGISTRY`.
- `risk_aversion`: Passed to the utility function to adjust curvature.
- `returns`: Output from `calculate_returns`, used as input to utility.
# Models

## `Model`
### Purpose

This module defines the abstract base class `Model`, which serves as the interface for all agent learning algorithms within the simulation framework. Its main goal is to allow for flexible and extensible integration of different learning behaviors—such as Genetic Algorithms, Subjective Bayesian strategies, or Artificial Neural Networks—by enforcing a standard callable interface for all custom implementations.
### Custom Option Requirements

To create a custom learning algorithm, users must subclass `Model` and implement the `__call__` method. The method must take a `jnp.ndarray` of agents and return a `jnp.ndarray` of updated agent states (or actions), potentially using additional arguments via `*args` and `**kwargs`.

### Required:
- Inherit from `Model`
- Define a `label` attribute for identification
- Implement the `__call__` method with the following signature:

```python
def __call__(self, agents: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    ...
```

## `Bayesian`

### Purpose

The `bayesian` module implements a Bayesian learning algorithm for agents that estimate asset value from observed market trades. Agents update their prior beliefs about the asset price using observed trade data, applying conjugate normal updating rules. The updated beliefs affect their demand and trading behavior in future generations.

This model is flexible enough to support both **informed** and **uninformed** agents:
- Informed agents observe the full set of market trades.
- Uninformed agents observe only a random subset, sized by `uninformed_base_ratio`.
- Agents may also optionally apply post-update adjustments using a registered `information_policy`.

### Function Signatures

```python
class Bayesian(Model):

    @staticmethod
    def __call__(agents: jnp.ndarray) -> jnp.ndarray
```
- Routes learning behavior to either `_informed` or `_uninformed` based on the global `informed` flag.
- Informed agents update priors using all trades from the market.
- Uninformed agents update using randomly sampled trade sets.
- If an agent uses an `information_policy`, it is applied to further adjust its updated beliefs.
- If simulation does not include information advantages then all agents are given full market data.
---
```python
def update_priors(agents: jnp.ndarray, trades: jnp.ndarray) -> jnp.ndarray
```
- Vectorized function that updates each agent’s belief distribution using Bayesian conjugate normal updating:

$$\mu_n = \frac{\mu/\sigma^2 + \sum q_i p_i / \tau^2}{1/\sigma^2 + \sum q_i / \tau^2}, \quad \sigma_n^2 = \left(\frac{1}{\sigma^2} + \frac{\sum q_i}{\tau^2}\right)^{-1}$$
- Input:
    - `agents[:, demand_fx_params]` contains prior mean (`μ`), std dev (`σ`), and signal precision (`τ`).
    - `trades` shape: `(n_agents, n_trades, 2)`.
---
### Used Components

- `learning_algorithm`: Must be set to `2` for Bayesian agents.
- `demand_fx_params`: Expected to store three values — prior mean (`μ`), standard deviation (`σ`), and signal precision (`τ`).
- `informed`: Used to partition agents and determine what trade data is visible to them.
- `information_policy`: Optional; references a function in `INFORMATION_POLICY_REGISTRY` to further alter post-update beliefs.
- `globals.trades`: Trade history used for inference, injected from the global simulation context.

### Allowed Function Pairings

| Function type      | Allowed versions                   |
| ------------------ | ---------------------------------- |
| Demand             | `BayesianDemand`                   |
| Spread             | `BayesianSpread`                   |
| Objective          | All                                |
| Utility            | All                                |
| Loss               | N/A                                |
| Information Policy | `BayesianInfo`, `ThompsonSampling` |
## `Genetic_Algorithm`

### Purpose

The `genetic_algorithm` module defines a `GeneticAlgorithm` learning model that simulates evolutionary adaptation across agent generations. It is one of the core learning systems in the framework and applies selection, crossover, and mutation operators to evolve agent parameters based on fitness.

Agents using this model are assumed to share their demand function structure, and operate using full access to others' fitness and parameter values—making it most suitable for baseline comparisons or specific experiments where agent internals are transparent.

### Function Signatures

```python
class GeneticAlgorithm(Model):
    label: str = "Genetic Algorithm"

    def __init__(self, agents: jnp.ndarray)

    def __call__(self, agents: jnp.ndarray) -> jnp.ndarray
```

#### Key Methods

- `__call__`: Applies either `informed_agents` or `uninformed_agents` based on `globals.informed`.
- `informed_agents(...)`: Runs the full GA pipeline—selection, crossover, mutation—for informed agents.
- `uninformed_agents(...)`: Same as above, but tailored for agents without signal-based information.
- `select_individual(...)`: Performs fitness-proportional (roulette wheel) selection.

### Genetic Algorithm Details

- **Selection**: Proportional to (positive-shifted) fitness.
- **Crossover**: One-point crossover at midpoint of parameter vector.
- **Mutation**: Adds small Gaussian noise to parameter values with probability `mutation_rate`.
- **Failsafe**: Handles possible `IndexError` in uneven populations.
### Used Components

- `learning_algorithm`: Must be set to `1` to select this learning strategy.
- `fitness`: Drives selection probability.
- `demand_fx_params`: Modified during crossover/mutation.
- `informed`: Directs the model to use appropriate evolution logic.
- `config.mutation_rate`, `config.crossover_rate`: Must be set for GA to function.
### Allowed Function Pairings


| Function type      | Allowed versions     |
| ------------------ | -------------------- |
| Demand             | `GS_linear`          |
| Spread             | `LinearDemandSpread` |
| Objective          | All                  |
| Utility            | All                  |
| Loss               | N/A                  |
| Information Policy | N/A                  |

## `Neural_Network`
### Purpose

The `neural_network` module defines a learning model based on TensorFlow neural networks. Agents using this model train individual networks to map observed prices to optimal behavior (e.g., maximizing utility or returns). The model supports both informed and uninformed agents, and optionally integrates information policies and custom loss functions.

To manage memory usage, trained models can either be held in memory or saved to disk depending on the `config.memory_optimization` setting.

### Function Signatures

```python
class NeuralNetwork(Model):
    def __init__(self, agents: jnp.ndarray)

    def __call__(nn_learners: jnp.ndarray, *args, **kwargs) -> jnp.ndarray
```
#### Internal Methods

- `_build_model(params)`: Constructs a feedforward network based on architecture parameters.
- `_load_model(model_ref)`: Loads a model from file path or returns in-memory reference.
- `_save_model(...)`: Saves a trained model and clears memory if needed.
- `_prepare_training_data()`: Extracts `X_train, y_train` from full trade history.
- `_prepare_uninformed_training_data()`: Samples a reduced set of trade data for uninformed agents.
- `_informed(...)` / `_uninformed(...)`: Apply training procedure to agent subset.
- `info_policy(...)`: Optionally post-processes agent state using a registered `InformationDecisionPolicy`.
### Used Components

- `learning_algorithm`: Must be set to `3` to select this model.
- `agent_id`: Unique identifier for the agent
- `informed`: Flag for whether agent is informed or not.
- `demand_fx_params`: Used as input features and network output.
- `info_params`: Parameters passed to `InformationDecisionPolicy` constructor.
- `information_policy`: Dictates whether/how the agent uses an info policy wrapper.
- `loss`: Index into `LOSS_REGISTRY`, which provides a loss function instance.
- `learning_params`: Defines architecture and training details:
    - `input_shape`, `hidden_layers`, `hidden_nodes`
    - `optimizer`, `epochs`, `loss`, `optimization_steps`
### Notes on Training Flow

1. Each agent maintains a unique neural network.
2. Trade data is split into inputs (`price`) and targets (e.g., utility or demand-related outcome).
3. Models are compiled using agent-specific loss/optimizer settings.
4. Informed agents use all trade data, while uninformed agents use a random subset.

### Memory Optimization

If `config.memory_optimization` is enabled:
- Models are saved to `.keras` files per agent.
- Keras sessions and Python references are cleared using `gc.collect()` after saving.

### Allowed Function Pairings

| Function type      | Allowed versions                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| Demand             | N/A                                                                                                                 |
| Spread             | `LinearDemandSpread`, N/A                                                                                           |
| Objective          | All, Requires a loss function that calculates the negative fitness                                                  |
| Utility            | All                                                                                                                 |
| Loss               | All, must calculate the negative fitness using the quantity as predictions and return to that trade as true values. |
| Information Policy | `ReinforcementLearning`, `BayesianInfo`, `ThompsonSampling`                                                         |
## `Loss_Functions`

### Purpose

The `loss` module defines a standardized interface for evaluating agent-specific fitness within neural network models. Loss functions here represent the inverse of expected utility (i.e., lower is better), and are compatible with TensorFlow's training APIs.

Each custom loss function should subclass `AgentLoss`, a lightweight wrapper around `tf.keras.losses.Loss`, and implement both `call()` and `get_config()` to support serialization and memory optimization workflows.

These losses are typically used in neural network agents, where the model predicts optimal demand and the loss measures alignment with economic goals like utility maximization.

### Custom Option Requirements

To create a custom loss function:
1. Subclass `AgentLoss`.
2. Assign a unique `name` and `reduction` strategy.
3. Implement the `call(y_true, y_pred)` method.
4. Provide a `get_config()` method for TensorFlow serialization.
5. Register the class using `register_loss(...)`.

#### Example

```python
class MyLoss(AgentLoss):
    name = "my_loss"
    reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred) ** self.alpha)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(alpha=self.alpha)
        return cfg

register_loss(MyLoss)
```
---
### Function Signatures

```python
class AgentLoss(tf.keras.losses.Loss, metaclass=ABCMeta):
    name: str
    reduction: tf.keras.losses.Reduction

    @abstractmethod
    def call(self, y_true, y_pred)
    @abstractmethod
    def get_config()
```
---
```python
class NegCARA(AgentLoss):
    name = "neg_cara"
    reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

    def call(self, y_true, y_pred)
    def get_config()
```
- Implements the negative Constant Absolute Risk Aversion utility:
    $$\mathbb{E}[e^{-\text{risk\_aversion} \cdot (\text{quantity} \cdot \text{return})}]$$
- Returns the **mean disutility** over the batch.
### Registry

Losses are selected in neural network agents using the `loss` index in `learning_params`. The currently available loss functions are:

|Key|Loss Function|
|---|---|
|1|NegCARA|

New loss functions are appended sequentially and must be serializable via `get_config()`.
### Used Components

- `loss`: Integer that maps to a function in `LOSS_REGISTRY`.
- `loss_params`: Passed as initialization arguments to the selected loss function.
- `learning_algorithm`: Must be set to use a model that supports loss-based optimization (e.g., Neural Network).
## `Information_Policy`

### Purpose

The `information_policy` module defines a family of decision-making strategies agents can use to choose whether to become informed each generation. These policies govern dynamic information acquisition and allow agents to conditionally pay for signal access based on expected rewards, learned behavior, or fixed roles.

All policies subclass `InformationDecisionPolicy` and are selected using the `information_policy` component of each agent.

### Custom Option Requirements

To define a custom information policy:
1. Subclass `InformationDecisionPolicy`.
2. Assign a unique `name`.
3. Implement the `__call__` method.
4. Register the policy using `register_info_policy(...)`.

#### Example

```python
class MyInfoPolicy(InformationDecisionPolicy):
    name = "MyPolicy"

    @staticmethod
    def __call__(agents: jnp.ndarray) -> jnp.ndarray:
        # Modify agents[:, informed] based on a custom rule
        return agents

register_info_policy(MyInfoPolicy)```
---
### Function Signatures

```python
class InformationDecisionPolicy(ABC):
    name: str

    @abstractmethod
    def __call__(*args, **kwargs)
```
---
#### Implemented Policies

##### `FixedInformation`

- Leaves the agent’s `informed` value unchanged across all generations.
- Used when informed status is set statically in the agent CSV.
##### `BayesianInfo`

- Agents update expected informed/uninformed utility based on past fitness.
- Then draw informed status from a Bernoulli with:
    $$p = \frac{\mathbb{E}[\text{informed return}]}{\mathbb{E}[\text{informed return}]+\mathbb{E}[\text{uninformed return}]}$$
##### `ReinforcementLearning`

- Learns a policy over informed/uninformed actions using softmax logits and entropy-regularized rewards.
- Uses TensorFlow and updates weights based on fitness performance.
- Must be instantiated

##### `ThompsonSampling`

- Details will be provided once implementation is complete, currently identical to BayesianInfo
### Registry

Agents select a policy using the `information_policy` component. The current options are:

| Key | Policy Name           |
| --- | --------------------- |
| 0   | FixedInformation      |
| 1   | BayesianInfo          |
| 2   | ReinforcementLearning |
| 3   | ThompsonSampling      |

Custom policies are assigned new keys in registration order.
### Used Components

- `informed`: The flag toggled by the policy to mark whether the agent has price signal access.
- `fitness`: Used as the reward signal in adaptive or RL-based policies.
- `info_params`: Stores expected fitness under informed/uninformed status for Bayesian and Thompson policies.
- `globals.generation`: Used in averaging reward estimates and RL update schedules.
- `agent_id`: Used by the neural network model to associate policy instances with agents.