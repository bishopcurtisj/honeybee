import random

def fitness(x):
    """
    Fitness function to maximize.
    Here, we simply use x^2.
    """
    return x ** 2

# GA parameters
POP_SIZE = 20        # Number of individuals in the population
MUTATION_RATE = 0.1  # Probability of mutating an offspring
CROSSOVER_RATE = 0.7 # Probability of performing crossover
GENERATIONS = 50     # Number of generations to evolve

# 1. Initialization: create an initial population of random values in [-10, 10]
population = [random.uniform(-10, 10) for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    # 2. Evaluate fitness for each individual
    fitnesses = [fitness(ind) for ind in population]
    
    # 3. Selection: use a "roulette wheel" style selection
    total_fitness = sum(fitnesses)
    
    def select_individual():
        """Select an individual from the population with probability proportional to its fitness."""
        r = random.uniform(0, total_fitness)
        running_sum = 0
        for ind, fit in zip(population, fitnesses):
            running_sum += fit
            if running_sum >= r:
                return ind
        # Fallback (should rarely happen if rounding issues occur)
        return population[-1]
    
    new_population = []
    
    # 4. Crossover + 5. Mutation: create new offspring
    while len(new_population) < POP_SIZE:
        parent1 = select_individual()
        parent2 = select_individual()
        
        # Perform crossover with some probability
        if random.random() < CROSSOVER_RATE:
            # Single arithmetic crossover
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
        else:
            # No crossover -> clone the parents
            child1, child2 = parent1, parent2
        
        # Mutation step: slightly perturb some offspring
        if random.random() < MUTATION_RATE:
            child1 += random.gauss(0, 1)  # add small random noise
        if random.random() < MUTATION_RATE:
            child2 += random.gauss(0, 1)
        
        new_population.append(child1)
        new_population.append(child2)
    
    # 6. Replacement: the new population becomes the current population
    population = new_population[:POP_SIZE]

# After all generations, find the best individual
best_individual = max(population, key=fitness)
print("Best individual:", best_individual)
print("Best fitness:", fitness(best_individual))





import jax
import jax.numpy as jnp
from jax import random, grad, jit

# Initialize random parameters for weights and biases
def init_params(layers, key):
    params = []
    keys = random.split(key, len(layers) - 1)
    for i in range(len(layers) - 1):
        w_key, b_key = random.split(keys[i])
        params.append((
            random.normal(w_key, (layers[i], layers[i+1])) * 0.1,
            random.normal(b_key, (layers[i+1],)) * 0.1
        ))
    return params

# Define the forward pass
def forward_pass(params, x):
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = jax.nn.relu(x)  # Apply ReLU activation
    # Output layer (no activation)
    final_w, final_b = params[-1]
    return jnp.dot(x, final_w) + final_b

# Loss function: Mean Squared Error
def mse_loss(params, x, y):
    preds = forward_pass(params, x)
    return jnp.mean((preds - y) ** 2)

# Update parameters using gradient descent
@jit
def update_params(params, x, y, lr):
    grads = grad(mse_loss)(params, x, y)
    updated_params = [
        (w - lr * dw, b - lr * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
    return updated_params

# Training loop
def train_network(params, x_train, y_train, epochs, lr):
    for epoch in range(epochs):
        params = update_params(params, x_train, y_train, lr)
        if epoch % 100 == 0:
            loss = mse_loss(params, x_train, y_train)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return params

# Example usage
if __name__ == "__main__":
    # Define network architecture
    layers = [2, 16, 1]  # Input size 2, one hidden layer with 16 units, output size 1
    key = random.PRNGKey(42)

    # Initialize parameters
    params = init_params(layers, key)

    # Generate some dummy data
    x_train = random.normal(key, (100, 2))  # 100 samples, 2 features each
    y_train = jnp.sum(x_train, axis=1, keepdims=True)  # Target is sum of inputs

    # Train the network
    trained_params = train_network(params, x_train, y_train, epochs=1000, lr=0.01)
