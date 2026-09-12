import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import polars as pl
import time

#cd Classifying-Fashion-Product-Text-and-Images

data = pd.read_csv('styles.csv', on_bad_lines='skip')

#performing eda
print('data head:', data.head(n=len(data)))
print('data info:', data.info())
print('data description:', data.describe())
print('missing values:', data.isnull().sum())

#to address missing vals, all columns that have str data types will be filled
strMissingVals = ['baseColour', 'season', 'usage', 'productDisplayName']
data[strMissingVals] = data[strMissingVals].fillna('Unknown') 
#only id and year have numeric data types and between them only year has missing vals 
data['year'] = data['year'].fillna(data['year'].mean())

#apply filters to extract meaningful subsets of data
print('handbag data:', data[data['articleType'] == 'Handbags'].head()) # i just like handbags
print('season counts by gender:', data.groupby('gender')['season'].value_counts()) #noticed mostly summer products across all genders
print('article type for women and girls:', data.groupby((data['gender'] == 'Women') | (data['gender'] == 'Girls'))['articleType'].value_counts())
#i might use this filter for my ml model  
baseColour_df = data.groupby(['gender', 'season', 'baseColour']).size()
print('baseColour_df:', baseColour_df)

#ml algorithm to classify and predict baseColour based on season and gender
X = pd.get_dummies(data[['gender', 'season']])
y = data['baseColour']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
result = model.predict(X_test)
print('Predicted base colours:', result)
print('Model accuracy:', model.score(X_test, y_test))

#data visualization from the model predictions
plt.figure(figsize=(10, 6))
sns.countplot(x=result)
plt.title('Predicted Base Colours Distribution')
plt.xlabel('Base Colour')
plt.ylabel('Count')
plt.show()

#data visualization of model accuracys
results_df = data.loc[X_test.index, ['gender', 'season']].copy()
results_df['correct'] = (y_test.values == result)
results_df.groupby(['season', 'gender'])['correct'].mean().unstack().plot(kind='bar')
plt.title('Model Accuracy by Season and Gender')
plt.ylabel('Accuracy')
plt.show()

#polars implementation
start = time.time()
pl_data = pl.read_csv('styles.csv', ignore_errors=True, truncate_ragged_lines=True)
pl_data = pl_data.with_columns([
    pl.col('baseColour').fill_null('Unknown'),
    pl.col('season').fill_null('Unknown'),
])
pl_grouped = pl_data.group_by(['gender', 'season', 'baseColour']).agg(pl.len().alias('count'))
polars_time = time.time() - start

#wanted to compare against pandas implementation  
start = time.time()
pd_data = pd.read_csv('styles.csv', on_bad_lines='skip')
pd_data['baseColour'] = pd_data['baseColour'].fillna('Unknown')
pd_data['season'] = pd_data['season'].fillna('Unknown')
pd_grouped = pd_data.groupby(['gender', 'season', 'baseColour']).size()
pandas_time = time.time() - start

print(f'pandas load and groupby time: {pandas_time:.4f} sec')
print(f'polars load and groupby time: {polars_time:.4f} sec')
print('polars grouped result (head):', pl_grouped.head())

plt.figure(figsize=(6, 5))
plt.bar(['Pandas', 'Polars'], [pandas_time, polars_time], color=['steelblue', 'darkorange'])
plt.ylabel('Seconds')
plt.title('Pandas vs Polars: Load and Groupby Runtime')
plt.show()