# Fashion Product Classification

Explores the Fashion Product Images (Small) dataset from Kaggle, does basic EDA with Pandas, and trains a simple classifier to predict clothing base colour from gender and season.

## Organization for Grading
Python script: classifier.py README file: this file, and Rust Jupyter notebook: rust_vs_python_intro.ipynb<br>
I want to clarify that I used the python template to create this, but only these 3 files are relevant for the assignment

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
