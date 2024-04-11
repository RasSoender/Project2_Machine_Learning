from MainClassification import *
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

N, M = X.shape

y = y.squeeze()

classNames = ["Over median Insulin Levels", "Under median Insulin Levels"]

C = len(classNames)


mu = np.mean(X, 0)
sigma = np.std(X, ddof=0)

X = (X - mu) / sigma


# Fit regularized logistic regression model to training data to predict
# the type of wine
lambda_interval = [0.083]
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

k = 0

mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])

mdl.fit(X, y)

w_est = mdl.coef_[0]
coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

print(w_est)
print(len(w_est))
print(len(attributeNames))
print(coefficient_norm) 


def coefficient_directions():
    plt.figure(figsize=(10, 6))
    colors = ['red' if coef < 0 else 'blue' for coef in w_est]
    plt.bar(attributeNames, w_est, color=colors)
    plt.title('Coefficients of Attributes')
    plt.xlabel('Attributes')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=70, ha='right')
    plt.tight_layout()
    plt.show()

coefficient_directions()