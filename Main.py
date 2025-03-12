import os
import numpy as np
import random as rn
from numpy import matlib
from AGTO import AGTO
from AOA import AOA
from FCM import FCM
from Global_Vars import Global_Vars
from MBO import MBO
from Model_A_SMU import Model_A_SMU
from Model_DenseNet import Model_DenseNet
from Model_Inception import Model_Inception
from Model_MD_3DHNet import Model_MD_3DHNet
from Model_MobileNet import Model_MobileNet
from Model_Res_Unet import Model_Res_Unet
from Model_Resnet import Model_RESNET
from Model_Trans_Unet import Model_Trans_Unet
from Model_UNET import Model_Unet
from Objective_Function import objfun_Segmentation
from Plot_results import *
from Proposed import PROPOSED
from RSO import RSO


# Read Dataset
an = 0
if an == 1:
    Dataset_path = './Dataset/Dataset/Data/'
    Path = os.listdir(Dataset_path)
    Orignal_Images = []
    Target = []
    count = 0
    for i in range(len(Path)):
        Folder = Dataset_path + Path[i]
        Fold_path = os.listdir(Folder)
        for j in range(len(Fold_path)):
            Classes = Folder + '/' + Fold_path[j]
            Class_path = os.listdir(Classes)
            for k in range(len(Class_path)):
                print(count)
                img_path = Classes + '/' + Class_path[k]
                image = cv.imread(img_path)
                image = cv.resize(image, (256, 256))

                Class_name = Classes.split('/')
                name = Class_name[5]
                if name == 'adenocarcinoma':
                    tar = 1
                elif name == 'large.cell.carcinoma':
                    tar = 2
                elif name == 'squamous.cell.carcinoma':
                    tar = 3
                elif name == 'normal':
                    tar = 0
                count = count + 1
                Orignal_Images.append(image)
                Target.append(tar)
    Uni = np.unique(Target)
    uni = np.asarray(Uni)
    Tar = np.zeros((len(Target), len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(Target == uni[j])
        Tar[ind, j] = 1
    np.save('Images.npy', Orignal_Images)
    np.save('Target.npy', Tar)

# Ground_Truth
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Seg_Img = []
    for j in range(len(Images)):
        print(j)
        image = Images[j]
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        med = cv.medianBlur(image, 3)  # Median Filter
        contra = cv.convertScaleAbs(med, alpha=alpha, beta=beta)  # Contrast Enhancement
        cluster = FCM(contra, image_bit=2, n_clusters=5, m=2, epsilon=0.8, max_iter=5)
        cluster.form_clusters()
        result = cluster.result.astype('uint8')
        uniq = np.unique(result)
        uniq = uniq[2:]
        lenUniq = [len(np.where(uniq[i] == result)[0]) for i in range(len(uniq))]
        index = np.argsort(lenUniq)
        img = np.zeros(image.shape, dtype=np.uint8)
        img[result == uniq[index[0]]] = 255
        kernel = np.ones((3, 3), np.uint8)
        img_erosion = cv.erode(img, kernel, iterations=1)
        img_dilate = cv.dilate(img_erosion, kernel, iterations=1)
        Seg_Img.append(img_erosion)
    np.save('Ground_Truth.npy', Seg_Img)

# optimization for Segmentation
an = 0
if an == 1:
    Feat = np.load('Images.npy', allow_pickle=True)  # Load the Selected features
    Target = np.load('Ground_Truth.npy', allow_pickle=True)  # Load the Target
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, No of epoches, Steps per epoch in MobileUnet
    xmin = matlib.repmat(np.asarray([5, 5, 300]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 1000]), Npop, 1)
    fname = objfun_Segmentation
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("MBO...")
    [bestfit1, fitness1, bestsol1, time1] = MBO(initsol, fname, xmin, xmax, Max_iter)  # MBO

    print("AOA...")
    [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("AGTO...")
    [bestfit4, fitness4, bestsol4, time3] = AGTO(initsol, fname, xmin, xmax, Max_iter)  # AGTO

    print("SCO...")
    [bestfit3, fitness3, bestsol3, time4] = RSO(initsol, fname, xmin, xmax, Max_iter)  # SCO

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
    np.save('Fittness.npy', fitness)
    np.save('BestSol_CLS.npy', BestSol_CLS)  # Bestsol classification

# Segmentation
an = 0
if an == 1:
    Data_path = './Original_images/'
    Data = np.load('Images.npy', allow_pickle=True)  # Load the Data
    Target = np.load('Ground_truth.npy', allow_pickle=True)  # Load the ground truth
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)  # Load the Bestsol Classification
    Unet = Model_Unet(Data_path)
    Res_Unet = Model_Res_Unet(Data, Target)
    Trans_Unet = Model_Trans_Unet(Data, Target)
    A_SMU = Model_A_SMU(Data, Target, [5, 5, 300])
    Proposed = Model_A_SMU(Data, Target, BestSol[4, 2:])
    Seg = [Unet, Res_Unet, Trans_Unet, A_SMU, Proposed]
    np.save('Segmented_image.npy', Proposed)
    np.save('Seg_img.npy', Seg)


# KFOLD - Classification
an = 0
if an == 1:
    EVAL_ALL = []
    Feat = np.load('Segmented_image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    K = 5
    Per = 1 / 5
    Perc = round(Feat.shape[0] * Per)
    eval = []
    for i in range(K):
        Eval = np.zeros((5, 14))
        Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
        Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Feat.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Feat[train_index, :]
        Train_Target = Target[train_index, :]
        Eval[0, :], pred_1 = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[1, :], pred_2 = Model_Inception(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[2, :], pred_3 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[3, :], pred_4 = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[4, :], pred_5 = Model_MD_3DHNet(Train_Data, Train_Target, Test_Data, Test_Target)
        eval.append(Eval)
    EVAL_ALL.append(eval)
    np.save('Eval_all_fold.npy', EVAL_ALL)


plot_conv()
PLot_ROC()
plot_results()
plot_results_kfold()
Images_Sample()
Image_segment()
