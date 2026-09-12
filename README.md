# Fashion Product Classification

Explores the Fashion Product Images (Small) dataset from Kaggle, does basic EDA with Pandas, and trains a simple classifier to predict clothing base colour from gender and season.

## Packages to Install
pandas
numpy
seaborn
matplotlib
scikit-learn

## Setup
python -m venv .venv
source .venv/bin/activate 
.venv\Scripts\activate # Windows
pip install -r requirements.txt

## Run
python classifier.py

## Example
Loads styles.csv, prints basic info (.head(), .info(), .describe(), missing values), filters/groups the data (e.g. season counts by gender), trains a Random Forest Classifier to predict baseColour from gender and season, and shows two plots: predicted colour distribution and model accuracy by season/gender.
