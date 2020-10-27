# Speaker Accent Detection using SVMs

This code reproduces the material presented in the report at the link [LINK]

It contains the following python files:
- `main.py`: main file, run this to produce full results in terms of summary tables and images.
- `nested_CV.py`: contains function for running nested cross validation.
- `traintest.py`: contains function for running full model selection, for both simple and Bagging model.

Additionally it contains the following GAMS file:
- `svm-accent-detection.gms`: GAMS model for SVM problem solution using project dataset.

`Data` folder contains the dataset for the problem:
- `accent-mfcc-data-1.csv`: original dataset in csv format.
- `accent-mfcc-data-1.xslsx`: excel file with data splitted in training and test set along with Kernel matrix.