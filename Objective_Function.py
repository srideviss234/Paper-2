import numpy as np
from Evaluation import net_evaluation
from Global_Vars import Global_Vars
from Model_A_SMU import Model_A_SMU


def objfun_Segmentation(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            predict = Model_A_SMU(Feat, Tar, sol)
            Eval = net_evaluation(predict, Tar)
            Fitn[i] = 1 / (Eval[4] + Eval[6])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        predict = Model_A_SMU(Feat, Tar, sol)
        Eval = net_evaluation(predict, Tar)
        Fitn = 1 / (Eval[4] + Eval[6])
        return Fitn