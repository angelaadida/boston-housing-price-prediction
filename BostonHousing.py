import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from ydata_profiling import ProfileReport
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyRegressor
import lazypredict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import learning_curve

data = pd.read_csv("C:\\Users\\Dell\\Desktop\\AAA 2024\\2024 Python Data Science\\Machine Learning\\BostonHousing.csv")

# profile = ProfileReport(data, title = "BostonHousing_Report", explorative=True)
# profile.to_file("report_BostonHousing.html")
# print(os.path.abspath("report_BostonHousing.html"))

# Basic Information about the Dataset
# print(f"First 5 rows of the dataset:\n{data.head()}\n")
# print(f"Summary of the dataset (info):\n{data.info()}\n")
# print(f"Shape of the dataset (rows, columns): {data.shape}\n")
# print(f"Descriptive statistics for numerical columns:\n{data.describe()}\n")
# print(f"Missing values in each column:\n{data.isnull().sum()}\n")
# print(f"Number of unique values in each column:\n{data.nunique()}\n")
# print(f"List of columns in the dataset:\n{data.columns}\n")

# # Step2: Train Test Split
# # Split by column
# target = "medv"
# x=data.drop(target,axis=1)
# y=data[target]
# # Split by row
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
#
# # Step 3: Data preprocessing
# num_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
#     ])
#
# chas = x_train['chas'].unique()
# ord_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OrdinalEncoder(categories=[chas])),
#     ])
#
# preprocessor = ColumnTransformer(transformers=[
#     ('num_feature', num_transformer, ['crim','zn',"indus","nox","rm","age","dis","rad","tax","ptratio","b","lstat"]),
#     ('ord_transformer', ord_transformer, ["chas"]),
#    ])
#
# # # Step 4: Algorithm Model building.
# # reg = Pipeline(steps=[
# #     ("preprocessor", preprocessor),
# #     ("model", LinearRegression())
# # ])
# # reg.fit(x_train, y_train)
# # y_predict = reg.predict(x_test)
# # for i, j in zip(y_test, y_predict):
# #     print("Actual: {}. Predict: {}".format(i, j))
#
# # # Step 5: Model evaluation metrics
# # 5.1 LinearRegression
# # reg = Pipeline(steps=[
# #     ("preprocessor", preprocessor),
# #     ("model", LinearRegression())
# # ])
# # reg.fit(x_train, y_train)
# # y_predict = reg.predict(x_test)
# # for i, j in zip(y_test, y_predict):
# #     print("Actual: {}. Predict: {}".format(i, j))
# # print("MAE: {}".format(mean_squared_error(y_test, y_predict)))
# # print("MSE: {}".format(mean_absolute_percentage_error(y_test, y_predict)))
# # print("R2: {}".format(r2_score(y_test, y_predict)))
# # Output
# # MAE: 24.455244753383372
# # MSE: 0.16957554962843055
# # R2: 0.666521435285575
#
# # # 5.2 RandomForestRegressor
# # reg = Pipeline(steps=[
# #     ("preprocessor", preprocessor),
# #     ("model", RandomForestRegressor(random_state=100))
# # ])
# # reg.fit(x_train, y_train)
# # y_predict = reg.predict(x_test)
# # print("MAE: {}".format(mean_squared_error(y_test, y_predict)))
# # print("MSE: {}".format(mean_absolute_percentage_error(y_test, y_predict)))
# # print("R2: {}".format(r2_score(y_test, y_predict)))
# # # Output
# # # MAE: 8.782521049019607
# # # MSE: 0.11429018760247782
# # # R2: 0.8802390839455373
#
# # # 5.2.1 GridSearchCV put outside RandomForest (choose the best combination parameters)
# # reg = Pipeline(steps=[
# #     ("preprocessor", preprocessor),
# #     ("model", RandomForestRegressor(random_state=100))
# # ])
# #
# # params = {
# #     "model__n_estimators": [50, 100, 200],
# #     "model__criterion": ["squared_error", 'absolute_error', 'poisson'],
# #     "preprocessor__num_feature__imputer__strategy": ['median', 'mean']
# #     # "max_depth": [None, 2, 5],
# #     # "min_samples_split": [2,5,10]
# # }
# # model = GridSearchCV(reg, param_grid=params, scoring='r2', cv=6,
# #                      verbose=2, n_jobs=6)
# # model.fit(x_train, y_train)
# # print(model.best_score_)
# # print(model.best_params_)
#
# # output
# # 0.8302162453641277
# # {'model__criterion': 'poisson', 'model__n_estimators': 200, 'preprocessor__num_feature__imputer__strategy': 'median'}
#
# # 2.2 RandomizedSearchCV (choosing randomly in GridSearch)
# # reg = Pipeline(steps=[
# #     ("preprocessor", preprocessor),
# #     ("model", RandomForestRegressor(random_state=100))
# # ])
# #
# # params = {
# #     "model__n_estimators": [50, 100, 200],
# #     "model__criterion": ["squared_error", 'absolute_error', 'poisson'],
# #     "preprocessor__num_feature__imputer__strategy": ['median', 'mean'],
# #     "model__max_depth": [None, 2, 5],
# #     "model__min_samples_split": [2,5,10]
# # }
# # model = RandomizedSearchCV(reg, param_distributions=params, scoring='r2', cv=6,
# #                      verbose=2, n_jobs=6, n_iter=30)
# # model.fit(x_train, y_train)
# # print(model.best_score_)
# # print(model.best_params_)
# # Output
# # 0.8272785468978371
# # {'preprocessor__num_feature__imputer__strategy': 'median', 'model__n_estimators': 100, 'model__min_samples_split': 5, 'model__max_depth': None, 'model__criterion': 'poisson'}
#
# # # 3. Lazy Predict (evaluation models same time)
# # reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# # models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# # print(models)
# # Output
# #                                   Adjusted R-Squared  R-Squared  RMSE  Time Taken
# # Model
# # GradientBoostingRegressor                    0.90       0.91  2.59        0.21
# # XGBRegressor                                 0.89       0.90  2.69        0.16
# # RandomForestRegressor                        0.88       0.89  2.80        0.46
# # DecisionTreeRegressor                        0.87       0.89  2.89        0.01
# # LGBMRegressor                                0.87       0.88  2.93        0.06
#
#
# # 5.3.1 GridSearchCV outside GradientBoostingRegressor (choose the best combination of parameters)
# reg = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("model", GradientBoostingRegressor())
# ])
#
# # Define the parameter grid for GridSearchCV
# params = {
#     "model__n_estimators": [50, 100, 200],
#     "model__criterion": ["squared_error", "absolute_error", "poisson"],
#     "preprocessor__num_feature__imputer__strategy": ["median", "mean"]
#     # "model__max_depth": [None, 2, 5],
#     # "model__min_samples_split": [2, 5, 10]
# }
# # Perform GridSearchCV to find the best parameters
# model = GridSearchCV(reg, param_grid=params, scoring='r2', cv=6,
#                      verbose=2, n_jobs=-1)
# # Fit the model
# model.fit(x_train, y_train)
# # Print the best score and best parameters found by GridSearchCV
# print("Best Score:", model.best_score_)
# print("Best Parameters:", model.best_params_)
# # # output
# # Best Score: 0.8567213403553309
# # Best Parameters: {'model__criterion': 'squared_error', 'model__n_estimators': 200, 'preprocessor__num_feature__imputer__strategy': 'mean'}
#
# # # Step 6: Feature Importance Analysis
# # # Get feature importance from the best model
# # feature_importances = model.best_estimator_.named_steps['model'].feature_importances_
# # num_features = x_train.select_dtypes(include=['float64', 'int64']).columns
# # for feature, importance in sorted(zip(num_features, feature_importances), key=lambda x: x[1], reverse=True):
# #     print(f"{feature}: {importance:.2f}")
# # # Plot Feature Importances
# # plt.bar(x_train.columns, feature_importances)
# # plt.xlabel('Feature')
# # plt.ylabel('Importance')
# # plt.title('Feature Importances')
# # plt.xticks(rotation=90)
# # plt.show()
#
# # Step 7: Learning Curves (Optional)
# train_sizes, train_scores, test_scores = learning_curve(
#     reg, x_train, y_train, cv=6, scoring='r2', n_jobs=-1
# )
# # Plot Learning Curves
# plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
# plt.plot(train_sizes, test_scores.mean(axis=1), label='Test score')
# plt.xlabel('Training Size')
# plt.ylabel('R² Score')
# plt.title('Learning Curves')
# plt.legend()
# plt.show()
#
# # # Step 8: Model Interpretation (Optional, using SHAP for GradientBoostingRegressor)
# # explainer = shap.TreeExplainer(model.best_estimator_.named_steps['model'])
# # shap_values = explainer.shap_values(x_train)
# # # Plot SHAP Summary
# # shap.summary_plot(shap_values, x_train)
#
# # # Step 9: Save the Model (Optional for future use or deployment)
# # joblib.dump(model.best_estimator_, 'gradient_boosting_model.pkl')
# # # Print a confirmation message
# # print("Model has been saved successfully!")