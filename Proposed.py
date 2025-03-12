## Rat Swarm Optimizer (RSO)
import numpy as np
import time


def PROPOSED(Positions, objective, Lower_bound, Upper_bound, Max_iterations):
    Search_Agents, dimension = Positions.shape
    Score = np.inf
    fitness = np.zeros(Search_Agents)
    for i in range(Search_Agents):
        fitness[i] = objective(Positions[i, :])
    Convergence = np.zeros((Max_iterations, 1))
    l = 0
    x = 1
    y = 5
    R = int(np.floor(np.multiply((y - x), np.random.rand(1, 1)) + x))
    ct = time.time()

    while l < Max_iterations:
        for i in range(Positions.shape[1 - 1]):
            Flag4Upper_bound = Positions[i, :] > Upper_bound[i, :]
            Flag4Lower_bound = Positions[i, :] < Lower_bound[i, :]
            Positions[i, :] = (np.multiply(Positions[i, :], (~ (Flag4Upper_bound + Flag4Lower_bound)))) + np.multiply(
                Upper_bound[i, :], Flag4Upper_bound) + np.multiply(Lower_bound[i, :], Flag4Lower_bound)

            fitness[i] = objective(Positions[i, :])
            if fitness[i] < Score:
                Score = fitness[i]
                Position = Positions[i, :]
        # Proposed update
        R = np.min(fitness) / (np.sqrt((np.min(fitness)) ** 2 + (np.max(fitness)) ** 2))
        A = R - l * ((R) / Max_iterations)
        for i in range(Positions.shape[1 - 1]):
            for j in range(Positions.shape[2 - 1]):
                C = 2 * np.random.rand()
                P_vec = A * Positions[i, j] + np.abs(C * ((Position[j] - Positions[i, j])))
                P_final = Position[j] - P_vec
                Positions[i, j] = P_final

        Convergence[l] = Score
        l += 1
    ct = time.time() - ct
    return Score, Convergence, Position, ct

