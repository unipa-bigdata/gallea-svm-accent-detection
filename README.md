# Speaker Accent Detection using SVMs

This code reproduces the material presented in the report at the link [https://github.com/unipa-bigdata/gallea-svm-accent-detection/blob/master/speaker_accent_detection_using_SVMs.pdf](https://github.com/unipa-bigdata/gallea-svm-accent-detection/blob/master/speaker_accent_detection_using_SVMs.pdf)

It contains the following python files:
- `main.ppynb`: notebook file, run this to produce full results in terms of summary tables and images.
- `main.py`: main file, run this to produce full results in terms of summary tables and images.
- `nested_CV.py`: contains function for running nested cross validation.
- `traintest.py`: contains function for running full model selection, for both simple and Bagging model.
- `explain.py`: contains function for running model explaination using SHAP framework.

Additionally it contains the following GAMS file:
- `svm-accent-detection.gms`: GAMS model for SVM problem solution using project dataset.

`Data` folder contains the dataset for the problem:
- `accent-mfcc-data-1.csv`: original dataset in csv format.
- `accent-mfcc-data-1.xslsx`: excel file with data splitted in training and test set along with Kernel matrix.