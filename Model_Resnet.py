import numpy as np
import cv2 as cv
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Dense
from Evaluation import evaluation


def Model_RESNET_FEAT(Data, Tar):
    IMG_SIZE = [124, 124, 3]
    Feat1 = np.zeros((Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Data.shape[0]):
        # gray=cv.cvtColor(Data[i].astype(np.uint8), cv.COLOR_BGR2HSV)
        Feat1[i, :] = cv.resize(Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    base_model.add(Dense(units=Tar.shape[1], activation='sigmoid'))   #  units=Tar.shape[1], activation='sigmoid'
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    try:
        base_model.fit(Data, Tar)
        f2 = base_model.layers[-1].get_weights()
    except:
        f2 = base_model.layers[-1].get_weights()  # Dense Layer
    Feat = cv.resize(f2[0], (f2[0].shape[0], Tar.shape[0]))
    return Feat


def Model_RESNET(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = 1

    IMG_SIZE = [224, 224, 3]
    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    train_data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    test_data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    act = ['Relu', 'linear', 'tanh', 'sigmoid']
    activation = ['Relu', 'linear', 'tanh', 'sigmoid']
    base_model.add(Dense(units=int(sol), activation='linear'))  # units=train_target.shape[1]
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    pred = base_model.predict(test_data)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred

