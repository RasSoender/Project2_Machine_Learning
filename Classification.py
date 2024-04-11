from MainClassification import *
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

def baseline_error(X_train, y_train, X_test, y_test):
    
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    # Make predictions
    y_est = dummy_clf.predict(X_test)

    error = np.sum(y_est!=y_test)/y_test.shape[0]

    return error


def train_KNN(neighbors, X_train, y_train, X_test, y_test):
    errors = []
    for n in neighbors: 
        knclassifier = KNeighborsClassifier(n_neighbors=n)
        knclassifier.fit(X_train, y_train)
        y_est = knclassifier.predict(X_test)

        errors.append(np.sum(y_est!=y_test)/y_test.shape[0])
        
    min_error = min(errors)
    n = neighbors[errors.index(min_error)]

    return n, min_error


def train_regression(lambda_interval, X_train, y_train, X_test, y_test):
    """
    training and testing the model for linear regression
    """

    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, ddof=0)

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])

        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T

        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
        #print(f"test_error = {test_error_rate[k]} - lambda = {lambda_interval[k]}")

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]

    return opt_lambda, min_error

# Preparing the data
N, M = X.shape

y = y.squeeze()
# Add offset attribute

K1 = K2 = 10

CV = model_selection.KFold(K1, shuffle=True)

# Values of lambda
lambdas = np.logspace(-10, 9, 50)

neighbors = list(range(1,20))

# Define outer CV
outer_cv = KFold(n_splits=K1, shuffle=True)
inner_cv = KFold(n_splits=K2, shuffle=True)

k_outer = 1
# Modify the main loop to use the train_regression function
for train_outer_index, test_outer_index in outer_cv.split(X, y):
    X_train_outer = X[train_outer_index]
    y_train_outer = y[train_outer_index]
    X_test_outer = X[test_outer_index]
    y_test_outer = y[test_outer_index]

    best_error_KNN = float('inf')
    best_error_Reg = float('inf')
    best_neighbors = None
    best_lambda = None

    k_inner = 1
    for train_inner_index, test_inner_index in inner_cv.split(X_train_outer, y_train_outer):
        X_train_inner = X_train_outer[train_inner_index]
        y_train_inner = y_train_outer[train_inner_index]
        X_test_inner = X_train_outer[test_inner_index]
        y_test_inner = y_train_outer[test_inner_index]

        # # Train ANN
        n, E_knn = train_KNN(neighbors, X_train_inner, y_train_inner, X_test_inner, y_test_inner)

        # # Train logistic regression
        lambda_log_reg, E_log_reg = train_regression(lambdas, X_train_inner, y_train_inner, X_test_inner, y_test_inner)

        if E_log_reg < best_error_Reg:
            best_error_Reg = E_log_reg
            best_lambda = lambda_log_reg

        if E_knn < best_error_KNN:
            best_error_KNN = E_knn
            best_neighbors = n

        k_inner += 1

    # Train models on the optimal hyperparameters
    _, outer_fold_knn_error = train_KNN([best_neighbors], X_train_outer, y_train_outer, X_test_outer, y_test_outer)
    _, outer_fold_reg_error = train_regression([best_lambda], X_train_outer, y_train_outer, X_test_outer, y_test_outer)

    # Compute baseline error
    baseline_error_outer = baseline_error(X_train_outer, y_train_outer, X_test_outer, y_test_outer)

    # Print results for the outer fold
    print(f"Outer Fold Nr. {k_outer}")
    print(f"KNN: n = {best_neighbors}, error = {outer_fold_knn_error}")
    print(f"Regression: lambda = {best_lambda}, error = {outer_fold_reg_error}")
    print(f"Baseline: {baseline_error_outer}")

    k_outer += 1