from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, BaggingClassifier,\
                                AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

# models dict with (estimator: hyperparams) key-value pairs
models = {
    RandomForestClassifier(): {
        'n_estimators': [10, 35, 70, 100],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 0.1, 0.2, 0.5],
        'min_samples_leaf': [1, 2, 0.2, 0.5],
        'max_features': [1, 2, None],
        'random_state': [42]
    },

    SGDClassifier(): {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        'random_state': [42]
    },

    HistGradientBoostingClassifier(): {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'max_iter': [35, 70, 100],
        'max_depth': [3, 5, 7, 10],
        'l2_regularization': [0, 0.2, 0.5],
        'min_samples_leaf': [10, 20, 40, 100],
        'random_state': [42]
    },

    BaggingClassifier(): {
        'n_estimators': [10, 30, 50, 60],
        'max_samples': [0.1, 0.25, 0.5, 0.7, 1],
        'max_features': [0.2, 0.5, 1, 2],
        'random_state': [42]
    },

    AdaBoostClassifier(): {
        'estimator': [DecisionTreeClassifier(max_depth=10)],
        'n_estimators': [30, 50, 70],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'random_state': [42]
    },

    # Deep learning but estimator definition fits more here in ML script vs DL script
    MLPClassifier(): {
        'hidden_layer_sizes': [(64,), (128,), (256,)],
        'alpha': [0.0001, 0.001, 0.01],
        'batch_size': [64, 128, 256],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300, 400, 500],  # hasnt converged with <=100
        'random_state': [42]
    }
}
