## Rat Swarm Optimizer (RSO)
import numpy as np
import time


def RSO(Positions, objective, Lower_bound, Upper_bound, Max_iterations):
    Search_Agents, dimension = Positions.shape
    Position = np.zeros((1, dimension))
    Score = np.inf
    # Positions = init(Search_Agents, dimension, Upper_bound, Lower_bound)
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

            fitness = objective(Positions[i, :])
            if fitness < Score:
                Score = fitness
                Position = Positions[i, :]
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


def init(SearchAgents, dimension, upperbound, lowerbound):
    Boundary = upperbound.shape[2 - 1]
    if Boundary == 1:
        Pos = np.multiply(np.random.rand(SearchAgents, dimension), (upperbound - lowerbound)) + lowerbound

    if Boundary > 1:
        for i in np.arange(1, dimension + 1).reshape(-1):
            ub_i = upperbound(i)
            lb_i = lowerbound(i)
            Pos[:, i] = np.multiply(np.random.rand(SearchAgents.shape[1 - 1], 1), (ub_i - lb_i)) + lb_i

    return Pos