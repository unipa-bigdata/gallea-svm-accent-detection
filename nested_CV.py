from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np


def nestedCv(df, y_col, binary = False):
    X = np.array(df.drop(columns=[y_col], axis=1).to_numpy())

    y = np.array(df[[y_col]].to_numpy().ravel())

    # Number of random trials
    NUM_TRIALS = 30

    # Set up possible values of parameters to optimize over
    p_grid = {"C": [1, 10, 100],
              "gamma": [.01, .1]}

    # We will use a Support Vector Classifier with "rbf" kernel
    svm = SVC(kernel="rbf")

    # Arrays to store scores
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000, 10000]},
                        {'kernel': ['poly'], 'degree': [0.1, 0.5, 1, 2, 3, 4, 5, 6]},
                        {'kernel': ['sigmoid'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'coef0': [0, 0.01, 0.1, 1]},
                        {'kernel': ['linear']}]



    # Loop for each trial
    for i in range(NUM_TRIALS):
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(
            estimator=SVC(), param_grid=tuned_parameters, cv=inner_cv
        )
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

        inner_score = cross_val_score(clf, X=X, y=y, cv=inner_cv)
        innear_mean = inner_score.mean()
        inner_dev = inner_score.std()

        print(f"Trial {i+1} of {NUM_TRIALS}: mean score: {innear_mean:3f} score std.dev: {inner_dev:3f}")

        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']

        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # print()

    score_difference = non_nested_scores - nested_scores

    print("Average difference of {:6f} with std. dev. of {:6f}."
          .format(score_difference.mean(), score_difference.std()))

    # Plot scores on each trial for nested and non-nested CV
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
               ["Non-Nested CV", "Nested CV"],
               bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation on MFCC Dataset",
              x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
               ["Non-Nested CV - Nested CV Score"],
               bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")

    if binary:
        plt.savefig('simple-cv-vs-nested-cv-binary.png')
    else:
        plt.savefig('simple-cv-vs-nested-cv.png')

    plt.show()