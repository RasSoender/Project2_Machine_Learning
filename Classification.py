from MainClassification import *
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net

def baseline_error(baseline, y_test):
    error = 0
    
    # Determine the majority class in y_test by taking the sum of y_test and comparing it to the number of elements in y_test
    class_predict = 1 if np.sum(y_test) > (y_test.shape[0] / 2) else 0
    
    for i in y_test:
        if i != class_predict:
            error += 1
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
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])

        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T

        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = mdl.coef_[0]
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]

    return opt_lambda, min_error

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
        lambda_log_reg, E_log_reg = train_regression(lambdas, X_train_inner, y_train_inner, X_test_inner, y_test_inner)

        if E_log_reg < best_error_Reg:
            best_error_Reg = E_log_reg
            best_lambda = lambda_log_reg

        # if E_ann < best_error_ANN:
        #     best_error_ANN = E_ann
        #     best_hidden_units = h

        k_inner += 1

    # Train models on the optimal hyperparameters
    # _, outer_fold_ann_error = train_ANN([best_hidden_units], y, train_outer_index, test_outer_index)
    _, outer_fold_reg_error = train_regression([best_lambda], X_train_outer, y_train_outer, X_test_outer, y_test_outer)

    # Compute baseline error
    baseline_error_outer = baseline_error(y_train_outer, y_test_outer)

    # Print results for the outer fold
    print(f"Outer Fold Nr. {k_outer}")
    # print(f"ANN: h = {best_hidden_units}, error = {outer_fold_ann_error}")
    print(f"Regression: lambda = {best_lambda}, error = {outer_fold_reg_error}")
    print(f"Baseline: {baseline_error_outer}")

    k_outer += 1