import numpy as np
import time


# Mine Blast Optimization (MBO)
def MBO(agents, objective_func, min_bounds, max_bound, max_iter):
    # Initialize agents and their velocities
    [n_agents, dim] = agents.shape
    alpha = 0.2
    beta = 1.0
    gamma = 2.0
    # agents = np.random.uniform(min_bounds[:, 0], min_bounds[:, 1], size=(n_agents, len(min_bounds)))
    velocities = np.zeros_like(agents)

    # Evaluate initial objective function values
    obj_values = np.array([objective_func(agent) for agent in agents])

    # Initialize the best agent and objective function value
    best_agent_idx = np.argmin(obj_values)
    best_agent = agents[best_agent_idx]
    best_obj_value = obj_values[best_agent_idx]

    Convergence = np.zeros((max_iter, 1))
    ct = time.time()

    # Main loop
    for i in range(max_iter):
        # Calculate weights based on objective function values
        weights = obj_values - np.min(obj_values)
        weights = (weights / np.sum(weights))

        # Calculate new velocities and update agents
        for j in range(n_agents):
            r1 = np.random.random(size=min_bounds.shape[1])
            r2 = np.random.random(size=max_bound.shape[1])
            r3 = np.random.random(size=min_bounds.shape[1])
            velocities[j] = alpha * velocities[j] \
                            + beta * r1 * (best_agent - agents[j]) \
                            + gamma * r2 * (agents[j] - np.mean(agents, axis=0)) \
                            + gamma * r3 * (np.random.uniform(min_bounds[0, :], max_bound[1, :]) - agents[j])
            agents[j] = agents[j] + velocities[j]

            # Check if agent is within bounds
            for k in range(min_bounds.shape[1]):
                if agents[j, k] < min_bounds[j,k]:
                    agents[j, k] = min_bounds[0, k]
                    velocities[j, k] = -velocities[j, k]
                elif agents[j, k] > min_bounds[j, k]:
                    agents[j, k] = min_bounds[1, k]
                    velocities[j, k] = -velocities[j, k]

        # Evaluate objective function values
        obj_values = np.array([objective_func(agent) for agent in agents])

        # Update best agent and objective function value
        if np.min(obj_values) < best_obj_value:
            best_agent_idx = np.argmin(obj_values)
            best_agent = agents[best_agent_idx]
            best_obj_value = obj_values[best_agent_idx]
        Convergence[i] = best_obj_value
    ct -= time.time()

    return best_obj_value, Convergence, best_agent, ct
