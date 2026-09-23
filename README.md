# Fashion Product Classification

An EDA and machine learning project that predicts clothing base colour from gender and season using the Fashion Product Images dataset.

Explores the Fashion Product Images (Small) dataset from Kaggle, does basic EDA with Pandas, and trains a simple classifier to predict clothing base colour from gender and season.

## Organization for Grading
Python script: `classifier.py`
Test suite: `test_classifier.py`
README file: this file
Rust Jupyter notebook: `rust_vs_python_intro.ipynb`<br>
I want to clarify that I used the python template to create this, but only these files are relevant for the assignment

## File Structure
    ├── classifier.py              # main script: EDA, cleaning, modeling, visualization, pandas vs polars benchmark
    ├── test_classifier.py         # pytest suite covering classifier.py end to end
    ├── requirements.txt           # project dependencies
    ├── Makefile
    ├── .gitignore                 # ignores .venv/, __pycache__/, etc.
    ├── README.md                  # this file
    ├── rust_vs_python_intro.ipynb # Rust notebook, not part of this assignment
    └── styles.csv                 # dataset, download separately (see Dataset section)

## Dataset
Dataset: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle, `styles.csv`, a metadata table of clothing/accessory products with columns like `gender`, `baseColour`, `season`, `usage`, `articleType`, `productDisplayName`, and `year`.

Why this dataset: it's large enough to support real EDA and modeling but small enough to run locally without special hardware, and it mixes messy real-world fields (missing values, categorical columns, free-text product names) that make it a good fit for practicing both data cleaning and classification.

## Packages to Install
pandas
numpy
seaborn
matplotlib
scikit-learn
polars

## Setup
    python -m venv .venv
    source .venv/bin/activate
    .venv\Scripts\activate # Windows
    pip install -r requirements.txt

## Run
    python classifier.py

## Analysis Performed
1. **Data loading & cleaning**: loads `styles.csv`, skipping malformed rows, and fills missing values (`Unknown` for string columns, mean for `year`).
2. **Exploratory data analysis**: `.head()`, `.info()`, `.describe()`, missing-value counts, and grouped breakdowns (e.g. season counts by gender, article type for women/girls, base colour counts by gender and season).
3. **Modeling**: one-hot encodes `gender` and `season`, then trains a `RandomForestClassifier` to predict `baseColour` from those two features, evaluated with an 80/20 train-test split.
4. **Visualization**: bar chart of predicted colour distribution, and a chart of model accuracy broken down by season and gender.
5. **Pandas vs Polars benchmark**: times loading and grouping the same CSV in both libraries as a quick performance comparison.

<img width="1427" height="785" alt="Screenshot 2026-09-22 220044" src="https://github.com/user-attachments/assets/aeef2525-f892-464d-87a0-a68c403aae0a" />

## What I Learned / Found Interesting
* **Model accuracy**: using only `gender` and `season` to predict `baseColour` gives the model very little real signal. Overall accuracy landed around 0.23, and the model's predictions collapsed almost entirely to "Black" rather than discriminating between colours. Colour is driven far more by the product image and description than by who it's for or what time of year it is, so this was a good reminder that a model's ceiling is set by the features you feed it, not just the algorithm.
* **Signal strength**: accuracy by season/gender roughly tracked how skewed the colour distribution was within each group. Groups dominated by one or two common colours were easier for the model to "get right", for example Boys/Winter hit 1.0 accuracy, while more evenly mixed groups stayed in the 0.15 to 0.4 range.
* **Pandas vs Polars**: results here did not favor Polars the way I expected going in. In my run, pandas loaded and grouped the CSV in about 0.075 sec versus about 0.174 sec for Polars. At this dataset size the overhead of Polars' setup likely outweighs any parallelism benefit, so the "Polars is always faster" assumption did not hold for a simple load plus groupby on a small file.
* **Data quality**: `baseColour`, `season`, and `productDisplayName` had the most missing values in the raw data, while `gender` and `articleType` were essentially complete. The raw CSV also had malformed rows (a mismatch between expected and actual row counts during parsing), which is why `on_bad_lines='skip'` and `ignore_errors=True` were necessary rather than just a defensive habit. Worth keeping in mind before trusting any colour-based groupings too heavily.
