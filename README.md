# 🏠 Boston Housing Price Prediction

A Machine Learning regression project predicting **median house prices** in Boston suburbs. Multiple regression models are trained, compared, and tuned using GridSearchCV and RandomizedSearchCV.

---

## 📌 Target Variable

`medv` — Median value of owner-occupied homes in $1000s

---

## 📋 Features (13 input variables)

| Feature | Description |
|---------|-------------|
| crim | Per capita crime rate |
| zn | Proportion of residential land zoned for large lots |
| indus | Proportion of non-retail business acres |
| chas | Charles River dummy variable (1 if bounds river) |
| nox | Nitric oxide concentration |
| rm | Average number of rooms per dwelling |
| age | Proportion of owner-occupied units built before 1940 |
| dis | Distances to employment centers |
| rad | Accessibility to radial highways |
| tax | Property tax rate |
| ptratio | Pupil-teacher ratio |
| b | 1000(Bk - 0.63)² where Bk = proportion of Black residents |
| lstat | % lower status of the population |

---

## 📁 Project Structure

```
boston-housing-prediction/
├── BostonHousing.py       # Original source code
├── BostonHousing.csv      # Dataset
├── notebook.ipynb         # Full pipeline notebook
├── requirements.txt
├── README.md
├── project_summary.md
├── dataset_description.md
└── .gitignore
```

---

## ⚙️ Setup

```bash
git clone <repo-url>
cd boston-housing-prediction
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## 🚀 Run

```bash
# Via script
python BostonHousing.py

# Via notebook
jupyter notebook notebook.ipynb
```

---

## 🧠 Models & Pipeline

```
Raw Data
    ↓
Train/Test Split (80/20, random_state=42)
    ↓
Preprocessing Pipeline:
  Numerical: SimpleImputer(median) + StandardScaler
  Categorical (chas): SimpleImputer + OrdinalEncoder
    ↓
LazyPredict → Compare all sklearn models
    ↓
Train top 3 models:
  LinearRegression
  RandomForestRegressor
  GradientBoostingRegressor ✅ Best
    ↓
Hyperparameter Tuning:
  GridSearchCV (GradientBoosting)
  RandomizedSearchCV (RandomForest)
    ↓
Learning Curves
```

---

## 📊 Results

### Model Comparison

| Model | MSE | MAPE | R² |
|-------|-----|------|----|
| LinearRegression | 24.455 | 0.170 | 0.667 |
| RandomForestRegressor | 8.783 | 0.114 | 0.880 |
| GradientBoostingRegressor | **6.792** | **0.104** | **0.907** ✅ |

### GridSearchCV — GradientBoostingRegressor
```
Best Score  : 0.8567
Best Params : n_estimators=200, criterion=squared_error, imputer=mean
```

### RandomizedSearchCV — RandomForestRegressor
```
Best Score  : 0.8273
Best Params : n_estimators=100, criterion=poisson, min_samples_split=5, max_depth=None
```

### LazyPredict Top 5
```
GradientBoostingRegressor    R²: 0.91  RMSE: 2.59
XGBRegressor                 R²: 0.90  RMSE: 2.69
RandomForestRegressor        R²: 0.89  RMSE: 2.80
DecisionTreeRegressor        R²: 0.87  RMSE: 2.89
LGBMRegressor                R²: 0.87  RMSE: 2.93
```

---

## 📈 TensorBoard

Not applicable for this project (sklearn-based).

---

## 🔑 Techniques Used

- Exploratory Data Analysis (EDA)
- Missing value analysis with `missingno`
- Automated profiling with `ydata_profiling`
- Preprocessing Pipeline (`sklearn.pipeline`)
- `ColumnTransformer` for mixed feature types
- `LazyPredict` for rapid model comparison
- `GridSearchCV` for exhaustive hyperparameter search
- `RandomizedSearchCV` for efficient hyperparameter search
- Learning Curves for bias-variance analysis
- Model persistence with `joblib`
