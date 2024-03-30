import numpy as np
import pandas as pd
import os



"""
This file loads the data. It does one-out-of-K-encoding on the nomial attributes.

"""

directory_path = os.getcwd()

filename = directory_path + "/NHANES_age_prediction.csv"

df = pd.read_csv(filename)

raw_data = df.values


# We take the range from 1 instead of 0 so we can remove the ID's
cols = range(1, 10)
X = raw_data[:, cols]

# We found out the code shows that observation 414 contained corrupt data. We therefore removed it. 
# print(X[413, :])

X = np.delete(X, 413, axis=0)
raw_data = np.delete(raw_data, 413, axis=0)
    

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

classLabels = raw_data[:, 7] 
# Then determine which classes are in the data by finding the set of
# unique class labels
classNames = np.unique(classLabels)

classDict = dict(zip(classNames, range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])


# We have a lot of the categorical features such as age_group, gender, activity, diabetic.
# We will use "one-out-of-K-encoding" since it stops the model from any misleading assumptions and

# Deleting the attributes Senior & Adult
attributeNames = np.delete(attributeNames, [0])
X = np.delete(X, [0], axis=1)


# "one-out-of-K-encoding" for gender
gender = np.array(X[:, 1], dtype=int).T
K = len(np.unique(gender))

# Create a matrix of zeros with shape (number of elements in 'gender', K)
gender_encoding = np.zeros((gender.size, K))
# Use advanced indexing to set the corresponding element in each row to 1
gender_encoding[np.arange(gender.size), gender - np.min(gender)] = 1

X = np.concatenate((X[:, :1], X[:, 2:], gender_encoding), axis=1)
attributeNames = np.concatenate((attributeNames[:1], attributeNames[2:], ["Male", "Female"]), axis=0)


# "one-out-of-K-encoding" for activity

active = np.array(X[:, 1], dtype=int).T
K = len(np.unique(active))

# Create a matrix of zeros with shape (number of elements in 'gender', K)
active_encoding = np.zeros((active.size, K))
# Use advanced indexing to set the corresponding element in each row to 1
active_encoding[np.arange(active.size), active - np.min(active)] = 1

X = np.concatenate((X[:, :1], X[:, 2:], active_encoding), axis=1)
attributeNames = np.concatenate((attributeNames[:1], attributeNames[2:], ["Active", "Not Active"]), axis=0)

# "one-out-of-K-encoding" for diabetic

diabetic = np.array(X[:, 3], dtype=int).T
K = len(np.unique(diabetic))

diabetic_encoding = np.zeros((len(diabetic), K))

# Iterate through unique values and set the corresponding column to 1
for i, value in enumerate(np.unique(diabetic)):
    diabetic_encoding[:, i] = (diabetic == value).astype(int)


X = np.concatenate((X[:, :3], X[:, 4:], diabetic_encoding), axis=1)

attributeNames = np.concatenate((attributeNames[:3], attributeNames[4:], ["Diabetic", "Not Diabetic", "Borderline Diabetic"]), axis=0)



"""
A lot of the attributenames are not very explanatory so we will change them
"""  
attributeNames[0] = "Age"
attributeNames[1] = "BMI"
attributeNames[2] = "Blood Glucose"
attributeNames[3] = "Oral"
attributeNames[4] = "Insulin Levels"


C = len(classNames)

if X.dtype != float:
    X = X.astype(float)


# Taking my insulin level attribute that I want to predict
y = X[:, 4].reshape(-1, 1)

# Removing the insulin level attribute from the features
X = np.delete(X, 4, axis=1)


import matplotlib.pyplot as plt

attributeNames = ["Age", "BMI", "Blood Glucose", "Oral", "Male", "Female", "Active", "Not Active", "Diabetic", "Not Diabetic", "Borderline Diabetic"]
coefficient = [-2.39, 5.36, 0.3, 1.5, 0.25, -0.25, -0.18, 0.18, -0.17, -0.18, 0.32]

# Method to show the coefficients of the attributes
def coefficient_directions():
    plt.figure(figsize=(10, 6))
    colors = ['red' if coef < 0 else 'blue' for coef in coefficient]
    plt.bar(attributeNames, coefficient, color=colors)
    plt.title('Coefficients of Attributes')
    plt.xlabel('Attributes')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=70, ha='right')
    plt.tight_layout()
    plt.show()
