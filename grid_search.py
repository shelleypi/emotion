"""A script that performs exhaustive search over specified parameter values
for each estimator defined in hyperparams.py.

Pickles the classifiers with their best params determined by CV accuracy score."""

import pickle
from emotion_recognition import EmotionRecognizer
from hyperparams import models

# TODO: have to rerun bc we've put a duration/offset limitation on audio
if __name__ == '__main__':
    best_estimators = []
    for model, params in models.items():
        classifier = EmotionRecognizer(model, override=False)  # no need to generate the same datasets every time
        classifier.load_data()
        best_estimator, best_params, best_score = classifier.grid_search(params)

        best_estimators.append([best_estimator, best_params, best_score])
        print(f"{best_estimator} achieved {best_score:.2f} accuracy with params: {best_params}.")

    print("Pickling best classifiers...")
    with open("grid/best_classifiers.pickle", 'wb') as f:
        pickle.dump(best_estimators, f)
    print("Best classifiers pickled.")


# RandomForestClassifier(max_depth=10, max_features=2, random_state=42)
# achieved 0.54 accuracy
# with params: {'max_depth': 10, 'max_features': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}.

# SGDClassifier(alpha=0.001, penalty='elasticnet', random_state=42)
# achieved 0.45 accuracy
# with params: {'alpha': 0.001, 'penalty': 'elasticnet'}.

# HistGradientBoostingClassifier(l2_regularization=0.5, max_depth=7, min_samples_leaf=40, random_state=42)
# achieved 0.61 accuracy
# with params:
# {'l2_regularization': 0.5, 'learning_rate': 0.1, 'max_depth': 7, 'max_iter': 100, 'min_samples_leaf': 40}.

# BaggingClassifier(max_features=0.5, max_samples=0.7, n_estimators=60, random_state=42)
# achieved 0.59 accuracy
# with params: {'max_features': 0.5, 'max_samples': 0.7, 'n_estimators': 60}.

# AdaBoostClassifier(
# estimator=DecisionTreeClassifier(max_depth=10), learning_rate=0.01, n_estimators=70, random_state=42)
# achieved 0.55 accuracy
# with params:
# {'estimator': DecisionTreeClassifier(max_depth=10), 'learning_rate': 0.01, 'n_estimators': 70, 'random_state': 42}.

# MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(128,), random_state=42)
# achieved 0.60 accuracy
# with params: {'alpha': 0.01, 'batch_size': 256, 'hidden_layer_sizes': (128,),
#               'learning_rate': 'constant', 'max_iter': 200}.
