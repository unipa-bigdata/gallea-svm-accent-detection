import pandas as pd

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