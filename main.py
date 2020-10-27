import pandas as pd
from nested_CV import nestedCv
from traintest import trainAndTest

def loadData(filename):
    TrainDF = pd.read_excel(filename,
                            sheet_name='train')

    TestDF = pd.read_excel(filename,
                           sheet_name='test')

    return [TrainDF, TestDF]


def convertLabelsForBinaryClassification(TrainDF, TestDF, label):
    TrainDF_b = TrainDF.copy()
    mask_train = TrainDF['language'] != label
    TrainDF_b.loc[mask_train, 'language'] = f"NON-{label}"

    TestDF_b = TestDF.copy()
    mask_test = TestDF['language'] != label
    TestDF_b.loc[mask_test, 'language'] = f"NON-{label}"

    return [TrainDF_b, TestDF_b]


if __name__ == '__main__':
    [TrainDF, TestDF] = loadData('data/accent-mfcc-data-1.xlsx')


    # Create dataset for binary classification IT/NON-IT
    [TrainDF_b, TestDF_b] = convertLabelsForBinaryClassification(TrainDF, TestDF, label = "IT")

    # Perform nested cv for binary classification
    nestedCv(df=TrainDF_b, y_col="language", binary = True)

    # Perform training and testing for NOT normalized binary classification
    # without and with Bagging
    trainAndTest(TrainDF=TrainDF_b, TestDF=TestDF_b, y_col="language", normalize = False)

    # Perform training and testing for normalized binary classification
    # without and with Bagging
    trainAndTest(TrainDF=TrainDF_b, TestDF=TestDF_b, y_col="language", normalize = True)

    # Perform nested cv for multi-class classification
    nestedCv(df = TrainDF, y_col="language")

    # Perform training and testing for NOT normalized multi-class classification
    # without and with Bagging
    trainAndTest(TrainDF=TrainDF, TestDF=TestDF, y_col="language", normalize = False)

    # Perform training and testing for NOT normalized multi-class classification
    # without and with Bagging
    trainAndTest(TrainDF=TrainDF, TestDF=TestDF, y_col="language", normalize = True)


