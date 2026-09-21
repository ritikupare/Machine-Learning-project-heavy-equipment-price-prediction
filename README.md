# Heavy Equipment Selling Price Prediction

Predicting the auction selling price of heavy equipment using machine learning. Built for the [Heavy Equipment Selling Price Prediction Challenge](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge) on Kaggle.

## About the Competition

Auction houses sell thousands of pieces of heavy equipment every year — bulldozers, excavators, wheel loaders, motor graders and more. The goal of this competition is to predict the sale price of a machine given its characteristics such as equipment type, age, usage hours, size class and configuration details.

- **Type:** Regression
- **Metric:** RMSLE (Root Mean Squared Log Error)
- **Training data:** 138,701 auction records with 50 columns
- **Test data:** 15,000 records
- **Target:** `TargetValue` — the sale price in USD

All column names in the dataset are masked for this competition. I decoded them using the provided `metadata.csv` file to understand what each column actually represents.

## Approach

### 1. Exploratory Data Analysis

- Examined the overall structure — 50 columns, 44 categorical and 6 numeric
- Found two key data quality issues:
  - `ManufactureYear` contains a placeholder value of 1001 for unknown years (10.6% of rows)
  - `OperationalHoursMeter` has both missing values (40.8%) and zeros representing unknown readings
- Analysed target distribution — right-skewed, so log1p transform was applied to match the RMSLE metric
- Checked missingness patterns — 35 out of 50 columns have missing values, mostly structural (spec columns that don't apply to every machine type)
- Performed bivariate analysis to identify strong predictors — age vs price, equipment group vs price, size vs price

### 2. Feature Engineering

Created 11 new features from the raw data:

| Feature | Source | Purpose |
|---|---|---|
| `sale_year`, `sale_month`, `sale_quarter`, `sale_dow`, `sale_doy` | TransactionDate | Capture seasonal and temporal price patterns |
| `year_made_clean` | ManufactureYear | Cleaned version with 1001 placeholder removed |
| `age` | sale_year − year_made_clean | How old the machine was at auction — strongest numeric predictor |
| `hours` | OperationalHoursMeter | Cleaned usage hours (zeros treated as unknown) |
| `has_hours` | OperationalHoursMeter | Binary flag: 1 if a real reading exists, 0 if unknown |
| `hours_per_year` | hours / age | Usage intensity — same total hours means different wear depending on age |
| 5 frequency columns | High-cardinality categoricals | Count of how often each category appears in training data |

### 3. Preprocessing

- Dropped identifier columns (`TransactionID`, `AssetID`), near-empty columns (`col18`, `col19`), and the redundant `InventoryGroupCategory`
- Numeric columns: median imputation with missing-indicator flags
- Categorical columns: filled missing values with `__NA__` as its own category level
- Ordinal columns (`UtilizationTier`, `AssetScaleFactor`): encoded respecting the natural order
- All preprocessing fit on training data only — no data leakage

### 4. Model Building

Trained 5 models across 3 families to compare approaches:

| Model | Family | Validation RMSLE |
|---|---|---|
| Ridge Regression | Linear | 0.3995 |
| Random Forest | Bagging | 0.2136 |
| LightGBM | Boosting | 0.2082 |
| XGBoost | Boosting | 0.2070 |
| CatBoost | Boosting | 0.2245 |

### 5. Hyperparameter Tuning

Used RandomizedSearchCV on LightGBM (15 combinations, 3-fold CV) to search across 7 hyperparameters. Tuned validation RMSLE improved to **0.1997**.

### 6. Final Model

Two additional techniques pushed the score below the 0.20 cutoff:

- **Native categorical handling** — instead of ordinal encoding, categorical columns were cast to pandas category dtype so LightGBM can search for optimal category groupings at each split. This alone dropped RMSLE from 0.208 to ~0.196
- **Seed ensemble** — averaged predictions from models trained with different random seeds to reduce prediction variance

Final validation RMSLE: **~0.196**

## Repository Structure

```
├── notebook.ipynb          # Complete Kaggle notebook with all code and outputs
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── .gitignore              # Files excluded from version control
```

> **Note:** The competition dataset is not included in this repository. It can be downloaded from the [competition page](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge/data) on Kaggle (requires accepting competition rules).

## How to Run

1. Clone this repository
2. Download the competition data from Kaggle and place `train.csv`, `test.csv`, `metadata.csv` and `sample_submission.csv` in a `data/` folder
3. Open `notebook.ipynb` in Jupyter or upload it to Kaggle
4. Run all cells

Alternatively, view the notebook directly on Kaggle where it was developed and submitted.

## Tech Stack

- Python 3.12
- pandas, numpy — data manipulation
- matplotlib, seaborn — visualization
- scikit-learn — preprocessing pipelines, model evaluation, hyperparameter tuning
- LightGBM — final model and native categorical handling
- XGBoost, CatBoost — model comparison

## Key Learnings

- Log-transforming a skewed target to match the RMSLE metric directly optimizes what the competition scores
- Feature engineering from domain knowledge (age, has_hours flag, usage intensity) contributed more than hyperparameter tuning alone
- LightGBM's native categorical handling significantly outperformed ordinal encoding on high-cardinality columns
- Seed ensembling is a simple technique that reduces prediction variance with no new information needed
- Preprocessing must be fit only on training data to prevent data leakage — using sklearn Pipeline and ColumnTransformer enforces this automatically
