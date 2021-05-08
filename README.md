# Boosting for Transfer Learning

Python implementation of the classic transfer learninig [paper](https://dl.acm.org/doi/10.1145/1273496.1273521) [1]

## Main idea
The main purpose of the paper is to address the diff-distribution issue between the source data and the target data. The proposed framework handled in a way such that for the source instances, when they are wrongly predicted, these samples could be those that are the most dissimilar to the target instances. Therefore, there is a mechanism to decrease their weights to reduce the effects.

## Where to apply

- You have trained a model on the historical data and pushed the model into production.
Overtime, you have collected more live data which might be distrbuited differently from the historical data. However, there is a large overhead re-training the model so you probably want to consider leverage the transfer learning such as the *TrAdaBoost* to incorporate the new knowledge without re-training everything from scratch.
- You want to build a model on the top of the union the two diff-distribution datasets which share the same feature/label data schema.

## Usage

### Basic usage
It requires that `X` is an input dataframe that is a concatenation of target dataframe and source dataframe, identified by the `domain_column`.
Currently this column needs to be hard-coded `source`, to denote the source domain sample otherwise it will be trated as samples from the target domain.


```python
from models import TrAdaBoostClassifier
from sklearn.svm import SVC


model = TrAdaBoostClassifier(base_estimator=SVC(), domain_column='domain')
model.fit(X=X, y=y)
model.predict(X=X)
```

### Hyper-parameter search in the base estimator 
```python
from models import TrAdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


xgboost_params = {
        'model_params': {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'random_state': 0
        },
        'tuning_params': {
            'param_distributions': {
                'eta': [0.05, 0.1],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'subsample': [0.9, 1.0],
                'colsample_bytree': [0.6, 0.9],
                'lambda': [2, 5, 8],
                'alpha': [2, 5, 8],
                'min_child_weight': [1, 2, 5],
                'n_estimators': [100, 200],
                'scale_pos_weight': [7, 13]
            },
            'cv': 3,
            'n_iter': 20,
            'scoring': 'roc_auc',
            'random_state': 2020,
            'n_jobs': 4
        }
    }

base_estimator = RandomizedSearchCV(XGBClassifier(**xgboost_params.get('model_params')),
                                        **xgboost_params.get('tuning_params'))

model = TrAdaBoostClassifier(base_estimator=base_estimator, n_iters=20, verbose=True)
model.fit(X=X, y=y, domain_column='domain')
model.predict(X=X, domain_column='domain')
```


### Reference
[1] Wenyuan Dai, Qiang Yang, Gui-Rong Xue, and Yong Yu. Boosting for transfer learning. *In Proceedings of the 24th international conference on Machine learning*, pages 193–200, 2007.