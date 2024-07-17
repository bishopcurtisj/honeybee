# Subjective Agent Based Model using Prezo for simulation

# Structure

## Agent

Agent objects must store their unique state; elements of this state can be broken down into objective and subjective. Objective elements are concrete and contain real values. An example would be an agent's financial position; their liquidity, investments and other holdings would all fall in this category. Subjective elements on the other hand represent the beliefs the agent has that influence their decisions. These elements would include things such as asset evaluation models, risk-preferences, and uncertainty estimates. 
For an agent trading options this breakdown may be suitable:

**Objective**
- Cash on hand
- Underlying holdings
- Common knowledge about the system and it's components i.e.
  - Option contract details
  - Historical distribution
  - Exercise and rebalancing frequency
  - Risk-free/Bond rate

**Subjective**
- Option pricing model i.e.
  - LSM
  - HMC
  - BSM
- Risk preference
- Future volatility estimates
- Prior distribution
- Likelihood function

## System

The system will set rules and states that apply to all agents. Examples include trading frequency, contract structure, and public information (common knowledge). The system may require component pieces to fulfill this role. In the case of a market for trading options and the underlying components may include:

- Option contract information and structure
- Frequency which trades may occur and portfolios may be rebalanced.
