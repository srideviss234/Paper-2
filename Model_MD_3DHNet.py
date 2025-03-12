from Evaluation import evaluation

from Model_DenseNet import Model_DenseNet
from Model_MobileNet import Model_MobileNet


def Model_MD_3DHNet(train_data, train_tar, test_data, test_tar):
    Eval_Dense, pred_dense = Model_DenseNet(train_data, train_tar, test_data, test_tar)
    Eval_Mob, pred_Mob = Model_MobileNet(train_data, train_tar, test_data, test_tar)
    pred = (pred_dense + pred_Mob) / 2
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_tar)
    return Eval

