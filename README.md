
# Bank Marketing – Machine Learning Project

## Project Overview

This project is part of our Machine Learning course at ESILV (A4 – Data & AI).

Our objective is to predict whether a customer will subscribe to a bank term deposit before being contacted during a marketing campaign.

The idea is to help banks identify customers who are more likely to subscribe instead of contacting everyone.

## Team Members

| Member | Responsibility |
|---|---|
| Chaima Garoui | EDA, data quality checks, interpretation and documentation |
| Youssef Damak | Data preprocessing, encoding and feature engineering |
| Mariem Dridi | Logistic Regression baseline and evaluation |
| Adam Nasri | Alternative models, hyperparameter tuning and comparison |

## Dataset

We are using the Bank Marketing dataset from the UCI Machine Learning Repository.

- File: `bank-additional-full.csv`
- Observations: 41,188
- Input features: 20
- Numerical features: 10
- Categorical features: 10
- Target variable: `y` (yes/no)
- ML task: Binary classification

Dataset source:
https://archive.ics.uci.edu/dataset/222/bank+marketing

## Exploratory Data Analysis (EDA)

The initial EDA has been completed. We explored the dataset structure, numerical distributions and subscription rates across different customer groups.

### Main Findings

| Data quality check | Result |
|---|---|
| Missing values (NaN) | 0 |
| Duplicate rows | 12 |
| Subscription (Yes) | 11.27% |
| No subscription (No) | 88.73% |
| Average customer age | 40.02 years |

Some categorical variables contain `unknown` values even though there are no standard missing values.

The target distribution is clearly imbalanced. This means that accuracy alone will not be enough to evaluate our models.

We also observed differences in subscription rates depending on age, occupation and education. These are exploratory findings and do not necessarily imply causal relationships.

## Preprocessing Recommendations

- Investigate the 12 duplicate rows before removing them.
- Handle `unknown` categorical values.
- Remove `duration` to prevent data leakage.
- Handle the special value 999 in `pdays`, which indicates that the customer was not previously contacted.
- Encode categorical variables.
- Scale numerical features where needed.
- Use stratified training, validation and test sets.
- Fit preprocessing transformations only on training data.

## Baseline Model

We plan to use Logistic Regression as our first predictive model, with a Dummy Classifier as a simple reference.

The models will be evaluated using precision, recall, F1-score, accuracy, ROC-AUC and Average Precision.

## Repository Files

- `01_EDA_and_Data_Quality.ipynb`: Exploratory data analysis and data quality checks.
- `bank-additional-full.csv`: Original dataset.
- `README.md`: Project documentation.

The EDA notebook loads the dataset directly from GitHub. No manual dataset upload is required.

## Project Progress

- [x] Problem definition
- [x] Dataset selection
- [x] Exploratory data analysis
- [x] Data quality analysis
- [x] Week 1 project specification
- [ ] Data cleaning and preprocessing
- [ ] Baseline implementation
- [ ] Initial model evaluation
- [ ] Alternative model experiments
- [ ] Final report and presentation

## Next Steps

The next step is to prepare the dataset for modelling. Once preprocessing is completed, we will train and evaluate the baseline model before experimenting with more advanced classifiers.
