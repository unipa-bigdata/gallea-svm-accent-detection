import numpy as np
import shap

shap.initjs()

from alibi.explainers import KernelShap
from sklearn import preprocessing
import matplotlib.pyplot as plt


def explain(model, TrainDF, TestDF, y_col, normalize=True):
    X_Train = np.array(TrainDF.drop(columns=[y_col], axis=1).to_numpy())

    y_Train = np.array(TrainDF[[y_col]].to_numpy())

    X_Test = np.array(TestDF.drop(columns=[y_col], axis=1).to_numpy())

    y_Test = np.array(TestDF[[y_col]].to_numpy())

    if normalize:
        X_Train = preprocessing.scale(X_Train)
        X_Test = preprocessing.scale(X_Test)

    model.fit(X_Train, y_Train.ravel())

    pred_fcn = model.decision_function
    np.random.seed(0)
    svm_explainer = KernelShap(pred_fcn)
    svm_explainer.fit(shap.kmeans(X_Train, 10))

    svm_explanation = svm_explainer.explain(X_Test, l1_reg=False)

    feature_names = TrainDF.columns.drop([y_col])

    classes = set(TrainDF[y_col])
    classes_it = iter(classes)

    for class_id in range(0, len(classes)):
        print(next(classes_it))
        plt.figure()
        shap.summary_plot(svm_explanation.shap_values[class_id], X_Test, feature_names, show=False)
        plt.savefig(f'shap_summary_plot_binary-{class_id}{"-normalized" if normalize else ""}.png')

    plt.figure()
    shap.summary_plot(svm_explanation.shap_values, X_Test, feature_names, show=False)
    plt.savefig(f'shap_summary_plot_binary_X{"-normalized" if normalize else ""}.png')
