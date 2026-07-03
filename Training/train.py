import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, 'Dataset', 'HDI.csv')
model_path = os.path.join(base_dir, 'Flask', 'HDI.pkl')

print("1. Importing the dataset...")
Development = pd.read_csv(dataset_path)

print("\n3. Selecting Dependent and Independent Variables...")
# We will create X directly without copying string type
X = pd.DataFrame()
le = LabelEncoder()
X['Country'] = le.fit_transform(Development.iloc[:, 2].astype(str))
X['Life expectancy'] = Development.iloc[:, 5]
X['Mean years of schooling'] = Development.iloc[:, 6]
X['Gross national income (GNI) per capita'] = Development.iloc[:, 7]
X['Internet users'] = Development.iloc[:, 67]

y = Development.iloc[:, 4].values
y = pd.DataFrame(y)

print("\n4. Checking and Handling Null Values...")
# Fill Null Values in X
X = X.fillna(X.mean())

print("\n5. Train and Test Data split...")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("\n6. Fit the Linear Regression Model...")
reg = LinearRegression()
reg.fit(x_train, y_train)

print("\n8. Saving the Model...")
with open(model_path, 'wb') as f:
    pickle.dump(reg, f)
    
print(f"Model successfully saved at {model_path}")
