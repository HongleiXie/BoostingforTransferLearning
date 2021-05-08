from sklearn.base import ClassifierMixin, BaseEstimator, clone
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_is_fitted
# from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.validation import _num_samples
from utility import check_X_domain_column
from typing import Any, Union, Tuple


class TrAdaBoostClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator: Any, n_iters: int = 10,
                 domain_column: str = 'domain', verbose: bool = False) -> None:

        assert getattr(base_estimator, 'fit', None) is not None
        assert getattr(base_estimator, 'predict', None) is not None
        assert isinstance(n_iters, int) and n_iters > 0
        # assert has_fit_parameter(base_estimator, 'sample_weight')

        self.base_estimator = base_estimator
        self.n_iters = n_iters
        self.verbose = verbose
        self.domain_column = domain_column

    def set_params(self, **params: Any) -> None:
        return super(TrAdaBoostClassifier, self).set_params(**params)

    def _normalize_weights(self, weights: np.array) -> np.array:
        return weights / np.sum(weights)

    def _calculate_error(self, y_true: Union[np.array, pd.Series], y_pred: np.array, weights: np.array) -> float:
        check_consistent_length(y_true, y_pred, weights)
        return np.sum(weights * np.abs(y_pred - y_true) / np.sum(weights))

    def initialize_parameters(self, n: int, m: int, init_weights: np.ndarray = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """
        n: size of source, diff-distribution
        m: size of target, same-distribution
        init_weights: Optional, initial weights
        Return initialized tuple of (init_weights, P, error, beta0, beta)
        """

        # initialize weights
        n_samples = n + m
        if init_weights is None:
            init_weights = np.ones(n_samples)
        else:
            assert _num_samples(init_weights) == n_samples

        P = np.empty((self.n_iters, n_samples))

        # initialize error vector
        error = np.empty(self.n_iters)
        beta0 = 1 / (1 + np.sqrt(2 * np.log(n / self.n_iters)))
        beta = np.empty(self.n_iters)

        return init_weights, P, error, beta0, beta

    def fit(self, X: pd.DataFrame, y: Union[np.array, pd.Series] = None,
            domain_column: str = None, init_weights: np.array = None) -> 'TrAdaBoostClassifier':

        # initialize data
        domain_column = domain_column if domain_column else self.domain_column
        check_X_domain_column(X, domain_column)
        # extract source domain and target domain data
        mask = X.loc[:, domain_column] == 'source'  # flag the source domain as 1 and target domain as 0
        X_ = X.drop(domain_column, axis=1, inplace=False)  # without the domain column
        X_, y = check_X_y(X_, y)

        # initialize parameters
        n = mask.sum()
        m = _num_samples(X_) - n
        weights, P, error, beta0, beta = self.initialize_parameters(n, m, init_weights)

        # initialize estimator list for each iteration
        estimators = []

        for t in np.arange(self.n_iters):
            P[t] = self._normalize_weights(weights)

            # Call learner
            est = clone(self.base_estimator).fit(X_, y, sample_weight=P[t])
            y_target_pred = est.predict(X_[~mask, :])

            # calculate the error on same-distribution data (X_same)
            error[t] = self._calculate_error(y[~mask], y_target_pred, weights[~mask])

            if self.verbose:
                print('Error_{}: {}'.format(t, error[t]))

            if error[t] > 0.5:
                error[t] = 0.5
                if self.verbose:
                    print('Cap the error to 0.5 as it is larger than 0.5 at round {}'.format(t))

            if error[t] == 0:
                self.n_iters = t
                beta = beta[:t]
                if self.verbose:
                    print('Early stopping at round {} because error is zero. Avoid over-fitting'.format(t))
                break

            beta[t] = error[t] / (1 - error[t])
            if self.verbose:
                print('beta_{}: {}'.format(t, beta[t]))

            # Update the new weight vector: for the source domain training instances, when
            # they are wrongly predicted, there is a mechanism to decrease the weights of these instances.
            if t < self.n_iters - 1:
                y_source_pred = est.predict(X_[mask, :])
                weights[mask] = weights[mask] * (beta0 ** np.abs(y_source_pred - y[mask]))  # source domain
                weights[~mask] = weights[~mask] * (beta[t] ** -np.abs(y_target_pred - y[~mask]))  # target domain

            estimators.append(est)

        if self.verbose:
            print('Number of iterations run: {}'.format(self.n_iters))

        self.fitted_ = True
        self.beta_ = beta
        self.estimators_ = estimators
        return self

    def predict(self, X: pd.DataFrame, domain_column: str = None) -> np.array:

        check_is_fitted(self, 'fitted_')

        # remove domain column if exists
        domain_column = domain_column if domain_column else self.domain_column

        X_ = X.drop(domain_column, axis=1, inplace=False,
                    errors='ignore')  # without the domain column, drop it only it exits

        size = X_.shape[0]  # size of the data to be predicted
        predicts = np.zeros(size)
        res = np.array([model.predict(X_.values) for model in self.estimators_])

        for i in range(size):
            right = np.sum(np.log(self.beta_[int(np.ceil(self.n_iters / 2)):]) * -0.5)
            left = np.sum(
                np.log(
                    self.beta_[int(np.ceil(self.n_iters / 2)):]
                ) * -1 * res[int(np.ceil(self.n_iters / 2)):][:, i]
            )
            predicts[i] = 1 if left >= right else 0

        return predicts

    def predict_proba(self, X: pd.DataFrame) -> None:
        raise NotImplementedError('predict the proba in TrAdaBoost is not implemented')
