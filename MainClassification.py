import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the dataset
directory_path = os.getcwd()
filename = directory_path + "/NHANES_age_prediction.csv"
df = pd.read_csv(filename)

raw_data = df.values

# We take the range from 1 instead of 0 so we can remove the ID's
cols = range(1, 10)
X = raw_data[:, cols]

# We found out the code shows that observation 414 contained corrupt data. We therefore removed it. 
X = np.delete(X, 413, axis=0)
raw_data = np.delete(raw_data, 413, axis=0)
    
# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

classLabels = raw_data[:, 7] 
# Then determine which classes are in the data by finding the set of unique class labels
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

# "one-out-of-K-encoding" for active
active = np.array(X[:, 1], dtype=int).T
K = len(np.unique(active))

# Create a matrix of zeros with shape (number of elements in 'active', K)
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

# Rename the attribute names
attributeNames[0] = "Age"
attributeNames[1] = "BMI"
attributeNames[2] = "Blood Glucose"
attributeNames[3] = "Oral"
attributeNames[4] = "Insulin Levels"

# Taking active and not active combining into one attribute as the target variable
# Check if column 6 is 1, set y to 1 for that row

median = np.median(X[:,4])

y[X[:, 4] > median] = 1

# Check if column 7 is 1, set y to 0 for that row
y[X[:, 4] <= median] = 0


# Removing the active and not active attribute from the features
X = np.delete(X, 4, axis=1)


# Define the attribute names again for plotting
attributeNames = ["Age", "BMI", "Blood Glucose", "Oral", "Male", "Female","Active", "Not Active", "Diabetic", "Not Diabetic", "Borderline Diabetic"]
