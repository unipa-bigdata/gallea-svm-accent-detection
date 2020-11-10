from nested_CV import nestedCv
from traintest import trainAndTest
from explain import explain
from utils import loadData, convertLabelsForBinaryClassification

if __name__ == '__main__':
    # Load dataset
    [TrainDF, TestDF] = loadData('data/accent-mfcc-data-1.xlsx')

    # Create dataset for binary classification IT/NON-IT
    [TrainDF_b, TestDF_b] = convertLabelsForBinaryClassification(TrainDF, TestDF, label="IT")

    # Perform nested cv for binary classification
    nestedCv(df=TrainDF_b, y_col="language", binary=True)

    # Perform training and testing for NOT normalized binary classification
    # without and with Bagging
    svm = trainAndTest(TrainDF=TrainDF_b, TestDF=TestDF_b, y_col="language", normalize=False)

    # Perform training and testing for normalized binary classification
    # without and with Bagging
    svm = trainAndTest(TrainDF=TrainDF_b, TestDF=TestDF_b, y_col="language", normalize=True)

    # Perform nested cv for multi-class classification
    nestedCv(df=TrainDF, y_col="language")

    # Perform training and testing for NOT normalized multi-class classification
    # without and with Bagging
    svm = trainAndTest(TrainDF=TrainDF, TestDF=TestDF, y_col="language", normalize=False)

    # Model explanation, very slow
    explain(model=svm, TrainDF=TrainDF, TestDF=TestDF_b, y_col="language", normalize=False)

    # Perform training and testing for NOT normalized multi-class classification
    # without and with Bagging
    svm = trainAndTest(TrainDF=TrainDF, TestDF=TestDF, y_col="language", normalize=True)

    # Model explanation, very slow
    explain(model=svm, TrainDF=TrainDF, TestDF=TestDF_b, y_col="language", normalize=True)
