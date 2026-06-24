# 📋 Project Summary — Boston Housing Price Prediction

## Overview
Regression project predicting **median house prices** (`medv`) in Boston suburbs using 13 features. The project follows a complete ML pipeline from EDA to hyperparameter tuning.

---

## Source File
| File | Description |
|------|-------------|
| BostonHousing.py | Original source code with full pipeline |

---

## ML Pipeline

### Step 1 — EDA
- Basic statistics, shape, dtypes
- Missing value analysis with `missingno`
- Automated profiling report with `ydata_profiling`
- Correlation heatmap
- Feature distributions

### Step 2 — Train/Test Split
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Train: 404 samples | Test: 102 samples
```

### Step 3 — Preprocessing Pipeline
```python
Numerical features (12):
  SimpleImputer(strategy='median') → StandardScaler()

Categorical feature (1 — chas):
  SimpleImputer(strategy='most_frequent') → OrdinalEncoder()

ColumnTransformer combines both
```

### Step 4 — LazyPredict
Compare all sklearn regressors automatically → Top: GradientBoosting

### Step 5 — Model Training
| Model | MSE | MAPE | R² |
|-------|-----|------|----|
| LinearRegression | 24.455 | 0.170 | 0.667 |
| RandomForestRegressor | 8.783 | 0.114 | 0.880 |
| GradientBoostingRegressor | — | — | ✅ Best |

### Step 6 — Hyperparameter Tuning
```
GridSearchCV (GradientBoosting, cv=6):
  Best Score: 0.8567
  Best: n_estimators=200, criterion=squared_error, imputer=mean

RandomizedSearchCV (RandomForest, cv=6, n_iter=30):
  Best Score: 0.8273
  Best: n_estimators=100, criterion=poisson, min_samples_split=5
```

### Step 7 — Learning Curves
Plot Train vs Validation R² score across different training sizes to detect overfitting/underfitting.

---

## Key Findings

- **`lstat`** (% lower status population) and **`rm`** (rooms per dwelling) have the strongest correlation with house price
- `chas` (Charles River) has almost no predictive power
- GradientBoosting consistently outperforms Linear Regression and Random Forest
- Model improves significantly with more data (learning curves show upward trend)

---

## Results

| Model | R² Score |
|-------|:--------:|
| LinearRegression | 0.667 |
| RandomForestRegressor | 0.880 |
| GradientBoostingRegressor (tuned) | **0.857** (CV) |
