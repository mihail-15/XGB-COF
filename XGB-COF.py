# import the necessary packages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor # import XGBRegressor 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import patheffects

# load the data from the 'xlsx' file (for example "pred_COF_E.xlsx")
data = pd.read_excel(r"C:\Users\kolev\OneDrive\1_БАН\0_Статии\23_Al-Si-Al2O3_Data-in-Brief\repo\Python\pred_COF_E.xlsx")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# define a function to perform hyperparameter tuning using GridSearchCV
def xgb_gridsearch(X_train, y_train, X_val, y_val):
 xgb = XGBRegressor(random_state=42) # use XGBRegressor 
 param_grid = {
 'eta': [0.1, 0.2, 0.4, 0.6], # use eta instead of max_features
 'n_estimators': [10, 20, 50, 100]
 }
 gs = GridSearchCV(xgb, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
 gs.fit(X_train, y_train)
 xgb_best = gs.best_estimator_
 y_val_pred = xgb_best.predict(X_val)
 r2 = r2_score(y_val, y_val_pred)
 rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
 mse = mean_squared_error(y_val, y_val_pred)
 mae = mean_absolute_error(y_val, y_val_pred)
 print("Best parameters: ", gs.best_params_)
 print("R2 score: {:.4f}".format(r2))
 print("RMSE: {:.4f}".format(rmse))
 print("MSE: {:.4f}".format(mse))
 print("MAE: {:.4f}".format(mae))
 return xgb_best

# perform hyperparameter tuning on the training and validation sets
xgb_best = xgb_gridsearch(X_train, y_train, X_val, y_val)

# train the XGB model on the entire training set using the best hyperparameters
xgb = XGBRegressor(learning_rate=xgb_best.learning_rate,n_estimators=xgb_best.n_estimators ,random_state=42) # use learning_rate instead of eta
xgb.fit(X_train,y_train)

# predict the coefficient of friction for the test set
y_pred = xgb.predict(X_test)

# predict the coefficient of friction for the validation set
y_val_pred = xgb.predict(X_val)

# calculate the R2 score for both sets
r2_test = r2_score(y_test,y_pred)
r2_val = r2_score(y_val,y_val_pred)

# calculate and store the performance metrics for both sets in a file (for example "E_performance_metrics.txt")
with open('E_performance_metrics.txt','w') as f:
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_test))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_test,y_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_test,y_pred)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_val))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_val,y_val_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_val,y_val_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_val,y_val_pred)))
 f.close()

# Shadow effect objects with different transparency and smaller linewidth
pe1 = [patheffects.SimpleLineShadow(offset=(0.5,-0.5), alpha=0.4), patheffects.Normal()]

# Plot of the actual vs predicted coefficient of friction as a function of time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend()

# x axis limit to 60
plt.xlim(0 ,450)

# y axis limit to 40
plt.ylim(0 ,0.6)

# gridlines to the plot
plt.grid(True)

plt.show()

fig = plt.figure() 
plt.plot(X,y)
plt.xlim(0 ,450)
plt.ylim(0 ,0.6)
plt.grid(True)

# Plot of the actual vs predicted coefficient of friction as a function of  time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend(loc='lower right')


# Save the plot with dpi=500 in 'png'
fig.savefig('pred_COF_E.png', dpi=500)




# create a DataFrame from the variables
df = pd.DataFrame({"Actual test": y_test, "Predicted test": y_pred, "Actual val": y_val, "Predicted val": y_val_pred})
# save the DataFrame to an Excel file
df.to_excel("test_val_data_E.xlsx", index=False)