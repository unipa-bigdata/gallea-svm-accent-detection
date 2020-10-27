import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

import matplotlib.pyplot as plt


def trainAndTest(TrainDF, TestDF, y_col, normalize = True):

    binary = len(set(TrainDF[y_col])) == 2

    X_Train = np.array(TrainDF.drop(columns=[y_col], axis=1).to_numpy())

    y_Train = np.array(TrainDF[[y_col]].to_numpy())

    X_Test = np.array(TestDF.drop(columns=[y_col], axis=1).to_numpy())

    y_Test = np.array(TestDF[[y_col]].to_numpy())

    if normalize:
        X_Train = preprocessing.scale(X_Train)
        X_Test = preprocessing.scale(X_Test)

    # Grid values for C and Gamma

    gridHyperPar = {"C": [10, 100, 1000, 10000],
                    "gamma": [0.00001, .0001, .001, .01, .1]}

    # Configure the outer cross-validation class

    crossValidationModel = KFold(n_splits=10)

    # svmModel = SVC(kernel="linear",cache_size=1000)

    svmModel = SVC(kernel='rbf',
                   decision_function_shape='ovr',
                   class_weight='balanced')


    gridSearch = GridSearchCV(svmModel,
                              gridHyperPar,
                              scoring='accuracy',
                              cv=crossValidationModel,
                              refit=True)

    searchResult = gridSearch.fit(X_Train,
                                  y_Train.ravel())

    best_model = searchResult.best_estimator_

    bestHyperParameterC = gridSearch.best_params_['C']
    bestHyperParameterGamma = gridSearch.best_params_['gamma']

    y_Predictions = best_model.predict(X_Test)

    score = accuracy_score(y_Test, y_Predictions)

    print('Single Model: Normalization = %r, Accuracy = %.3f, Best Hyper Parameter C = %.3f Best Hyper Parameter Gamma = %.5f' % (
        normalize, score, bestHyperParameterC, bestHyperParameterGamma))

    # Plot non-normalized confusion matrix

    print(classification_report(y_Test, y_Predictions))

    np.set_printoptions(precision=2)

    if (binary):
        plot_confusion_matrix(best_model, X_Test, y_Test)
        plt.savefig(f'SVM-Accent-confusion-binary{"-normalized" if normalize else ""}.png')
        plot_roc_curve(best_model, X_Test, y_Test)
        plt.savefig(f'SVM-Accent-roc-binary{"-normalized" if normalize else ""}.png')
        plot_precision_recall_curve(best_model, X_Test, y_Test)
        plt.savefig(f'SVM-Accent-precision-recall-binary{"-normalized" if normalize else ""}.png')
    else:
        plot_confusion_matrix(best_model, X_Test, y_Test)
        plt.savefig(f'SVM-Accent-confusion{"-normalized" if normalize else ""}.png')



    bagging_model = BaggingClassifier(base_estimator=gridSearch, n_estimators=30, random_state=314, n_jobs=4)\
        .fit(X_Train, y_Train.ravel())

    y_Predictions = bagging_model.predict(X_Test)

    score = accuracy_score(y_Test, y_Predictions)
    print('Bagging model: Normalization = %r, Accuracy = %.3f' % (normalize, score))
    print(classification_report(y_Test, y_Predictions))

    if (binary):
        plot_confusion_matrix(bagging_model, X_Test, y_Test)
        plt.savefig(f'SVM-Accent-confusion-binary-bagging{"-normalized" if normalize else ""}.png')
        plot_roc_curve(bagging_model, X_Test, y_Test)
        plt.savefig(f'SVM-Accent-roc-binary-bagging{"-normalized" if normalize else ""}.png')
        plot_precision_recall_curve(bagging_model, X_Test, y_Test)
        plt.savefig(f'SVM-Accent-precision-recall-binary-bagging{"-normalized" if normalize else ""}.png')
    else:
        plot_confusion_matrix(bagging_model, X_Test, y_Test)
        plt.savefig(f'SVM-Accent-confusion-bagging{"-normalized" if normalize else ""}.png')