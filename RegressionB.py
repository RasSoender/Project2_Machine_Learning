from Main import *
import torch
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net

def baseline_error(baseline, y_test):
    error = (
        np.square(y_test - baseline).sum(axis=0) / y_test.shape[0]
    )
    return error

def train_ANN(hidden, y):
    y = y.reshape(-1, 1)

    errors = []
    for h in hidden: 
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, h),  # M features to n_hidden_units
            torch.nn.ReLU(),  # 1st transfer function,
            torch.nn.Linear(h, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )
        loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
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
        y_test_est = net(X_test)

        # Determine errors and errors
        se = (y_test_est.float() - y_test.float()) ** 2  # squared error
        mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
        errors.append(mse)
    min_error = min(errors)
    h = errors.index(min_error)

    return h, min_error

def train_regression(X_train, y_train, X_test, y_test):
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

attributeNames = np.concatenate((attributeNames[:4], attributeNames[5:]), axis=0)
attributeNames = ["Offset"] + attributeNames.tolist()
M = M + 1


# Calculating baseline - mean(y)
baseline = np.mean(y)

K1 = K2 = 10

CV = model_selection.KFold(K1, shuffle=True)

# Values of lambda
lambdas = np.power(10.0, range(-5, 9))

# Values of lambda
hidden = list(range(1, 10))

k = 1
for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    h, E_ann = train_ANN(hidden, y)
    opt_lamb, E_lamb = train_regression(X_train, y_train, X_test, y_test)
    E_baseline = baseline_error(baseline, y_test)

    print(f"Fold Nr. {k}")
    print(f"h = {h}, error = {E_ann}")
    print(f"lamb = {opt_lamb}, error = {E_lamb}")
    print(f"baseline = {E_baseline}")

    k += 1