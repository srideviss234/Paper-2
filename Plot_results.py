import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
import cv2 as cv


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'MBO-A3D-SMUnet', 'AOA-A3D-SMUnet', 'AGTO-A3D-SMUnet', 'RSO-A3D-SMUnet', 'ERSO-A3D-SMUnet']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((5, 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[0, j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Dataset', 1, 'Statistical Report ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(50)
    Conv_Graph = Fitness[0]

    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=10,
             label='MBO-A3D-SMUnet')
    plt.plot(length, Conv_Graph[1, :], color='#89fe05', linewidth=3, marker='*', markerfacecolor='green',
             markersize=10, label='AOA-A3D-SMUnet')  # c
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='cyan',
             markersize=10, label='AGTO-A3D-SMUnet')
    plt.plot(length, Conv_Graph[3, :], color='#ffff14', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=10, label='RSO-A3D-SMUnet')  # y
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=10, label='ERSO-A3D-SMUnet')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence.png")
    plt.show()


def PLot_ROC():
    lw = 2
    cls = ['Resnet', 'Inception', 'MobileNet', 'Densenet', 'MD-3DHNet']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    colors = cycle(["blue", "crimson", "gold", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i], )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path1 = "./Results/ROC.png"
    plt.savefig(path1)
    plt.show()


def plot_results():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'MBO-A3D-SMUnet', 'AOA-A3D-SMUnet', 'AGTO-A3D-SMUnet', 'RSO-A3D-SMUnet', 'ERSO-A3D-SMUnet']
    SEG_CLS = ['TERMS', 'Unet', 'Res-Unet', 'Trans-Unet', '3D_SMUnet', 'ERSO-A3D-SMUnet']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    value_all = Eval_all[0, :]

    stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
    for i in range(4, value_all[0].shape[1] - 9):
        for j in range(value_all.shape[0] + 4):
            if j < value_all.shape[0]:
                stats[i, j, 0] = np.max(value_all[j][:, i])
                stats[i, j, 1] = np.min(value_all[j][:, i])
                stats[i, j, 2] = np.mean(value_all[j][:, i])
                stats[i, j, 3] = np.median(value_all[j][:, i])
                stats[i, j, 4] = np.std(value_all[j][:, i])
        X = np.arange(stats.shape[2])
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        ax.bar(X + 0.00, stats[i, 0, :], color='#fa4224', edgecolor='k', width=0.10, label="MBO-A3D-SMUnet")  # r  g  r
        ax.bar(X + 0.10, stats[i, 1, :], color='#b7fffa', edgecolor='k', width=0.10, label="AOA-A3D-SMUnet")  # g o
        ax.bar(X + 0.20, stats[i, 2, :], color='#0804f9', edgecolor='k', width=0.10, label="AGTO-A3D-SMUnet")  # b b
        ax.bar(X + 0.30, stats[i, 3, :], color='#21fc0d', edgecolor='k', width=0.10, label="RSO-A3D-SMUnet")  # m g
        ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='k', width=0.10, label="ERSO-A3D-SMUnet")  # k
        plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
        plt.xlabel('Statisticsal Analysis')
        plt.ylabel(Terms[i - 4])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Segmentation_%s_alg.png" % (Terms[i - 4])
        plt.savefig(path1)
        plt.show()

        X = np.arange(stats.shape[2])
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.bar(X + 0.00, stats[i, 5, :], color='#ff5b00', edgecolor='k', width=0.10, label="Unet")  # r
        ax.bar(X + 0.10, stats[i, 6, :], color='#08ff08', edgecolor='k', width=0.10, label="Res-Unet")  # g
        ax.bar(X + 0.20, stats[i, 7, :], color='#3d7afd', edgecolor='k', width=0.10, label="Trans-Unet")  # b
        ax.bar(X + 0.30, stats[i, 8, :], color='#ff0789', edgecolor='k', width=0.10, label="3D_SMUnet")  # m
        ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='k', width=0.10, label="ERSO-A3D-SMUnet")  # k
        plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
        plt.xlabel('Statisticsal Analysis')
        plt.ylabel(Terms[i - 4])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Segmentation_%s_met.png" % (Terms[i - 4])
        plt.savefig(path1)
        plt.show()


def plot_results_kfold():
    eval1 = np.load('Eval_all_fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 3, 4, 5, 9]
    Classifier = ['TERMS', 'Resnet', 'Inception', 'MobileNet', 'Densenet', 'MD-3DHNet']

    value1 = eval1[0, 4, :, 4:]
    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value1[j, :])
    print('-------------------------------------------------- 250 Epoch - Dataset', 1, 'Classifier Comparison',
          '--------------------------------------------------')
    print(Table)

    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval1.shape[1], eval1.shape[2]))
        for k in range(eval1.shape[1]):
            for l in range(eval1.shape[2]):
                if j == 9:
                    Graph[k, l] = eval1[0, k, l, Graph_Terms[j] + 4]
                else:
                    Graph[k, l] = eval1[0, k, l, Graph_Terms[j] + 4]

        X = np.arange(5)
        plt.plot(X, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label='Resnet')
        plt.plot(X, Graph[:, 1], color='#89fe05', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                 label='Inception')  # c
        plt.plot(X, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                 label='MobileNet')
        plt.plot(X, Graph[:, 3], color='#ad0afd', linewidth=3, marker='o', markerfacecolor='magenta', markersize=12,
                 label='Densenet')  # y
        plt.plot(X, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='black', markersize=12,
                 label='MD-3DHNet')
        plt.xticks(X + 0.10, ('50', '100', '150', '200', '250'))
        plt.xlabel('Epoch')
        plt.ylabel(Terms[Graph_Terms[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/%s_line_Epoch.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path1)
        plt.show()


def Images_Sample():
    Original = np.load('Images.npy', allow_pickle=True)
    for i in range(8, 9):
        # print(n, i)
        Orig_1 = Original[i]
        Orig_2 = Original[i + 1]
        Orig_3 = Original[i + 2]
        Orig_4 = Original[i + 3]
        Orig_5 = Original[i + 4]
        Orig_6 = Original[i + 5]
        plt.suptitle('Sample Images from Dataset', fontsize=25)
        plt.subplot(2, 3, 1).axis('off')
        plt.imshow(Orig_1)
        plt.subplot(2, 3, 2).axis('off')
        plt.imshow(Orig_2)
        plt.subplot(2, 3, 3).axis('off')
        plt.imshow(Orig_3)
        plt.subplot(2, 3, 4).axis('off')
        plt.imshow(Orig_4)
        plt.subplot(2, 3, 5).axis('off')
        plt.imshow(Orig_5)
        plt.subplot(2, 3, 6).axis('off')
        plt.imshow(Orig_6)
        plt.show()


def Image_segment():
    Original = np.load('Images.npy', allow_pickle=True)
    segmented = np.load('Seg_Img.npy', allow_pickle=True)
    Image = [406, 418, 428, 852, 856]
    for i in range(len(Image)):
        Orig = Original[Image[i]]
        Seg_1 = segmented[Image[i]]
        for j in range(1):
            # print(i, j)
            Orig_1 = Seg_1[j]
            Orig_2 = Seg_1[j + 1]
            Orig_3 = Seg_1[j + 2]
            Orig_4 = Seg_1[j + 3]
            Orig_5 = Seg_1[j + 4]
            plt.suptitle('Segmented Images from Dataset', fontsize=20)

            plt.subplot(2, 3, 1).axis('off')
            plt.imshow(Orig)
            plt.title('Orignal', fontsize=10)

            plt.subplot(2, 3, 2).axis('off')
            plt.imshow(Orig_1)
            plt.title('Unet', fontsize=10)

            plt.subplot(2, 3, 3).axis('off')
            plt.imshow(Orig_2)
            plt.title('Res-Unet', fontsize=10)

            plt.subplot(2, 3, 4).axis('off')
            plt.imshow(Orig_3)
            plt.title('Trans-Unet ', fontsize=10)

            plt.subplot(2, 3, 5).axis('off')
            plt.imshow(Orig_4)
            plt.title('3D_SMUnet', fontsize=10)

            plt.subplot(2, 3, 6).axis('off')
            plt.imshow(Orig_5)
            plt.title('ERSO-A3D-SMUnet', fontsize=10)
            plt.show()

            cv.imwrite('./Results/Image_results/Original_image_' + str(i + 1) + '.png', Orig)
            cv.imwrite('./Results/Image_results/segm_img_Unet_' + str(i + 1) + '.png', Orig_1)
            cv.imwrite('./Results/Image_results/segm_img_Res-Unet_' + str(i + 1) + '.png', Orig_2)
            cv.imwrite('./Results/Image_results/segm_img_Trans-Unet_' + str(i + 1) + '.png', Orig_3)
            cv.imwrite('./Results/Image_results/segm_img_3D Swin-based MobileUnet_' + str(i + 1) + '.png', Orig_4)
            cv.imwrite('./Results/Image_results/segm_img_PROPOSED_' + str(i + 1) + '.png', Orig_5)


if __name__ == '__main__':
    plot_conv()
    PLot_ROC()
    plot_results()
    plot_results_kfold()
    Images_Sample()
    Image_segment()
