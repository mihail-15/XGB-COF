# XGB-COF

XGB-COF is a python software that predicts the coefficient of friction (COF) of porous AlSi10Mg-Al2O3 composite materials using XGBoost, a powerful and scalable machine learning algorithm. The porous AlSi10Mg-Al2O3 composite materials were tested by pin-on-disk method under dry sliding conditions.

Requirements

The software requires python 3.7 or higher to run. The software also requires several packages that can be installed using pip or conda commands. The required packages are:

pandas: for data manipulation and analysis
numpy: for numerical computation
xgboost: for machine learning
sklearn: for machine learning
matplotlib: for visualization
patheffects: for adding effects to paths in matplotlib


Installation

The software can be downloaded from https://github.com/mihail-15/XGB-COF.git or https://codeocean.com/capsule/7130568/tree/v1. The software consists of one python script called XGB-COF.py that contains all the code for data loading, processing, modeling, evaluation, visualization, and saving. The software also requires an Excel file called pred_COF_AE.xlsx that contains the input and output variables for each sample of the porous AlSi10Mg-Al2O3 composite materials. Place the XLSX file in the same folder as the python code.

Usage

Open a terminal or command prompt window and go to the directory that has XGB-COF.py and pred_COF_AE.xlsx to run the software. Then type python XGB-COF.py and press enter. The software will start running and print some messages on the screen indicating its progress. The software will also create three files in the same directory: pred_COF_AE.png (the plot), AE_performance_metrics.txt (the performance metrics), and test_val_data_AE.xlsx (the actual and predicted COF data).

Output

The output of the software consists of three files:

pred_COF_AE.png: This file contains a plot of the actual vs predicted COF as a function of time for both test and validation sets using matplotlib. The plot illustrates the fit and accuracy of the XGBoost model.

AE_performance_metrics.txt: This file contains the performance metrics (R2, RMSE, MSE, MAE) of the XGBoost model on the test set. The fit and accuracy of the XGBoost model are assessed by these performance metrics.

test_val_data_AE.xlsx: This file contains a DataFrame from the actual and predicted COF values for both test and validation sets using pandas. This file can be used for further analysis or comparison of the results.

License

The software is licensed under the MIT license, See LICENSE for more details.
