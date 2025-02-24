from dataclasses import dataclass
import numpy as jnp
from typing import Protocol
import numpy.random as random

class LearningFunction(Protocol):
    def __call__(self) -> jnp.ndarray:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class GeneticAlgorithm:
    """ 
    Genetic Algorithm class to be used in the learning system
    Takes subset of agents with columns:
    [fitness, informed, demand_function_params...]
    """

    def __call__(self, agents: jnp.ndarray, parameters: jnp.ndarray, pop_size: int, informed: bool = True) -> jnp.ndarray:
        # Evaluate fitness for each individual
        fitnesses = agents[:,0].copy() 
        min_fitness = jnp.min(fitnesses)
        if min_fitness < 0:
            adjusted_fitnesses = fitnesses - min_fitness + 1e-6
        else:
            adjusted_fitnesses = fitnesses
        total_fitness = jnp.sum(adjusted_fitnesses)

        crossover_rate = parameters[0][0]
        mutation_rate = parameters[1][0]

        new_population = jnp.empty_like(agents)
        new_population[:,0] = agents[:,0]
        params = len(agents[0]) - 1
        crossover_point = params // 2

        # Crossover + Mutation: create new offspring
        try:
            for i in range(0, pop_size, 2):
                parent1 = self.select_individual(total_fitness=total_fitness, agents=agents, fitnesses=adjusted_fitnesses)[1:]
                parent2 = self.select_individual(total_fitness=total_fitness, agents=agents, fitnesses=adjusted_fitnesses)[1:]

                # Perform crossover with some probability
                if random.random() < crossover_rate:
                    child1 = jnp.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = jnp.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                else:
                    # No crossover -> clone the parents
                    child1, child2 = parent1, parent2
                
                if informed:
                    # Mutation step: slightly perturb some offspring
                    if random.uniform(0, 1) < mutation_rate:
                        child1[1:] += random.standard_normal(params-1)  # add small random noise
                    if random.uniform(0, 1) < mutation_rate:
                        child2[1:] += random.standard_normal(params-1)  # add small random noise
                else:  
                    if random.uniform(0, 1) < mutation_rate:
                        child1 += random.standard_normal(params)  # add small random noise
                    if random.uniform(0, 1) < mutation_rate:
                        child2 += random.standard_normal(params)  # add small random noise

                new_population[i, 1:] = child1
                new_population[i + 1, 1:] = child2
        except IndexError:
            pass

        return new_population

    def select_individual(self, total_fitness: float, agents: jnp.ndarray, fitnesses: jnp.ndarray) -> jnp.ndarray:
        """Select an individual from the population with probability proportional to its fitness."""
        try:
            r = random.uniform(0, total_fitness)
            running_sum = 0
            for ind, fit in zip(agents, fitnesses):
                running_sum += fit
                if running_sum >= r:
                    return ind
                
        # Fallback (should rarely happen if rounding issues occur)
            return agents[-1]
        except Exception as e:
            
            return agents[-1]

@dataclass(frozen=True, kw_only=True, slots=True)
class ThompsonSampler:
    def __call__(self) -> jnp.ndarray:
        ...
LEARNING_REGISTRY = {1: GeneticAlgorithm, 'GeneticAlgorithm': GeneticAlgorithm, 2: ThompsonSampler, 'ThompsonSampler': ThompsonSampler}

