from MainClassification import *
import torch
from sklearn.model_selection import KFold
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net

def baseline_error(baseline, y_test):
    error = 0
    numb_0 = np.count_nonzero(baseline == 0)
    numb_1 = np.count_nonzero(baseline == 1)
    print(numb_0)
    print(numb_1)
    
    # Determine the majority class in y_test by taking the sum of y_test and comparing it to the number of elements in y_test
    class_predict = 0 if np.sum(y_test) > y_test.shape[0] / 2 else 1
    
    for i in y_test:
        if i != class_predict:
            error += 1
    print(y_test.shape[0])
    print(error / y_test.shape[0])
    return error / y_test.shape[0]


def train_ANN(hidden, y, train_index, test_index):
    y = y.reshape(-1, 1)
    errors = []
    for h in hidden: 
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, h),  # M features to n_hidden_units
            torch.nn.ReLU(),  # 1st transfer function,
            torch.nn.Linear(h, 1),  # n_hidden_units to 1 output neuron
            torch.nn.Sigmoid()  # Sigmoid activation for binary classification
        )
        loss_fn = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index])
        max_iter = 10000
        n_replicates = 1
        # Train the net on training data
        net, _, _ = train_neural_net(
            model,
            loss_fn,
            X=X_train,
            y=y_train,
            n_replicates=n_replicates,
            max_iter=max_iter,
        )
        # Determine estimated class labels for test set
        y_test_est = (net(X_test) > 0.5).float()  # Threshold at 0.5 for binary classification

        # Determine accuracy as the evaluation metric
        accuracy = (y_test_est == y_test).float().mean().data.numpy()
        errors.append(1 - accuracy)  # Error is 1 - accuracy for binary classification
    min_error = min(errors)
    h = errors.index(min_error)

    return h, min_error


def train_regression(lambdas, X_train, y_train, X_test, y_test):
    """
    training and testing the model for linear regression
    """
    k = 0

    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    Error_train_rlr = np.empty((K, 1))
    Error_test_rlr = np.empty((K, 1))
    Error_train_nofeatures = np.empty((K, 1))
    Error_test_nofeatures = np.empty((K, 1))
    w_rlr = np.empty((M, K))
    mu = np.empty((K, M - 1))
    sigma = np.empty((K, M - 1))
    w_noreg = np.empty((M, K))

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, K2)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    error = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )
    return opt_lambda, error

# Preparing the data
N, M = X.shape

y = y.squeeze()
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

attributeNames = ["Offset"] + attributeNames
M = M + 1


# Calculating baseline - mean(y)
baseline = np.mean(y)

K1 = K2 = 10

CV = model_selection.KFold(K1, shuffle=True)

# Values of lambda
lambdas = np.power(10.0, range(-5, 9))

# Values of lambda
hidden = list(range(1, 10))

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

    best_error_ANN = float('inf')
    best_error_Reg = float('inf')
    best_hidden_units = None
    best_lambda = None

    k_inner = 1
    for train_inner_index, test_inner_index in inner_cv.split(X_train_outer, y_train_outer):
        X_train_inner = X_train_outer[train_inner_index]
        y_train_inner = y_train_outer[train_inner_index]
        X_test_inner = X_train_outer[test_inner_index]
        y_test_inner = y_train_outer[test_inner_index]

        # # Train ANN
        # h, E_ann = train_ANN(hidden, y_train_outer, train_inner_index, test_inner_index)

        # # Train logistic regression
        # E_log_reg = train_regression(X_train_inner, y_train_inner, X_test_inner, y_test_inner)

        # if E_log_reg < best_error_Reg:
        #     best_error_Reg = E_log_reg

        # if E_ann < best_error_ANN:
        #     best_error_ANN = E_ann
        #     best_hidden_units = h

        k_inner += 1

    # Train models on the optimal hyperparameters
    # _, outer_fold_ann_error = train_ANN([best_hidden_units], y, train_outer_index, test_outer_index)
    # outer_fold_reg_error = train_regression(X_train_outer, y_train_outer, X_test_outer, y_test_outer)

    # Compute baseline error
    baseline_error_outer = baseline_error(y_train_outer, y_test_outer)

    # Print results for the outer fold
    print(f"Outer Fold Nr. {k_outer}")
    # print(f"ANN: h = {best_hidden_units}, error = {outer_fold_ann_error}")
    # print(f"Logistic Regression: error = {outer_fold_reg_error}")
    print(f"Baseline: {baseline_error_outer}")

    k_outer += 1