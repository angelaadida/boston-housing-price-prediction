# 📦 Dataset — Boston Housing

## Overview

| Field | Detail |
|-------|--------|
| Dataset name | Boston Housing |
| Task | Regression (House Price Prediction) |
| Rows | 506 |
| Features | 13 input + 1 target |
| Target | medv (Median house value in $1000s) |
| Missing values | None |
| Source | Harrison & Rubinfeld (1978) |

---

## Features

| Feature | Type | Description |
|---------|------|-------------|
| crim | float | Per capita crime rate by town |
| zn | float | Proportion of residential land zoned for lots over 25,000 sq.ft |
| indus | float | Proportion of non-retail business acres per town |
| chas | int | Charles River dummy variable (1 if tract bounds river, 0 otherwise) |
| nox | float | Nitric oxides concentration (parts per 10 million) |
| rm | float | Average number of rooms per dwelling |
| age | float | Proportion of owner-occupied units built prior to 1940 |
| dis | float | Weighted distances to five Boston employment centers |
| rad | int | Index of accessibility to radial highways |
| tax | float | Full-value property-tax rate per $10,000 |
| ptratio | float | Pupil-teacher ratio by town |
| b | float | 1000(Bk - 0.63)² where Bk = proportion of Black residents |
| lstat | float | % lower status of the population |
| **medv** | **float** | **TARGET: Median value of owner-occupied homes in $1000s** |

---

## Key Statistics

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| medv | 5.0 | 50.0 | 22.5 | 9.2 |
| rm | 3.56 | 8.78 | 6.28 | 0.70 |
| lstat | 1.73 | 37.97 | 12.65 | 7.14 |
| crim | 0.006 | 88.98 | 3.61 | 8.60 |

---

## Top Correlations with Target (medv)

| Feature | Correlation |
|---------|-------------|
| lstat | **-0.74** (strong negative) |
| rm | **+0.70** (strong positive) |
| ptratio | -0.51 |
| indus | -0.48 |
| tax | -0.47 |
| chas | +0.18 (weak) |

---

## Download

- **Included in repo:** `BostonHousing.csv`
- **Kaggle:** https://www.kaggle.com/datasets/schirmerchad/bostonhoustingmlnd

---

## Load

```python
import pandas as pd
data = pd.read_csv('BostonHousing.csv')
print(data.shape)  # (506, 14)
```

---

## Notes

- No missing values in this dataset
- `chas` is binary (0/1) but treated as categorical in preprocessing
- `medv` is capped at 50.0 — some outliers at the top
- Dataset has mild skewness in features like `crim` and `b`
