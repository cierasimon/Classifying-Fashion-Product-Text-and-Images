import os
 
import matplotlib
matplotlib.use('Agg') #headless backend so plt.show() calls in the script dont block
 
import numpy as np
import pandas as pd
import pytest
import runpy
 
#tests for classifier.py, runs it with runpy.run_path() and checks the resulting vars
 
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'classifier.py')
 
 
@pytest.fixture
def script_namespace(tmp_path, monkeypatch):
    #build a small styles.csv, cd into that folder, then run the script and grab its vars
    df = pd.DataFrame({
        'id': range(1, 21),
        'gender': ['Men', 'Women', 'Women', 'Girls', 'Boys'] * 4,
        'baseColour': (['Black', 'White', 'Blue', 'Red'] * 4) + [np.nan] * 4,
        'season': ['Summer', 'Winter', 'Fall', 'Spring'] * 5,
        'usage': ['Casual'] * 20,
        'productDisplayName': [f'Product {i}' for i in range(20)],
        'year': [2015, 2016, np.nan, 2018] * 5,
        'articleType': ['Handbags', 'Shirts', 'Shoes', 'Jeans'] * 5,
    })
    df.to_csv(tmp_path / 'styles.csv', index=False)
    monkeypatch.chdir(tmp_path)
    return runpy.run_path(SCRIPT_PATH)
 
 
#check that the csv actually loaded into data with the columns we need
def test_data_loaded(script_namespace):
    data = script_namespace['data']
    assert len(data) == 20
    assert 'baseColour' in data.columns
 
 
#check that missing vals actually got filled during preprocessing
def test_missing_values_filled(script_namespace):
    data = script_namespace['data']
    assert data['baseColour'].isnull().sum() == 0
    assert data['year'].isnull().sum() == 0
 
 
#check the model actually trained and produced one prediction per test row
def test_model_trained_and_predicted(script_namespace):
    result = script_namespace['result']
    X_test = script_namespace['X_test']
    y_test = script_namespace['y_test']
    assert len(result) == len(X_test) == len(y_test)
 
 
#check model accuracy is a real score between 0 and 1
def test_model_accuracy_in_valid_range(script_namespace):
    model = script_namespace['model']
    X_test = script_namespace['X_test']
    y_test = script_namespace['y_test']
    accuracy = model.score(X_test, y_test)
    assert 0.0 <= accuracy <= 1.0
 
 
#system test: whole script runs end to end and produces real predictions
def test_full_script_runs_end_to_end(script_namespace):
    assert 'model' in script_namespace
    assert 'result' in script_namespace
    assert len(script_namespace['result']) > 0
    assert script_namespace['pandas_time'] >= 0
    assert script_namespace['polars_time'] >= 0