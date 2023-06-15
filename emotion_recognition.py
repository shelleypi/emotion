from data_preparation import prepare_data

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

class EmotionRecognizer:
    """
    A class for training and testing ML models on emotions based on
    features extracted from speech data done by AudioPreparation class in data_preparation.py.
    """
    def __init__(self, model=None, override=False):
        """
        :param sklearn model model: Model used to predict emotion.
            If None, when class instance calls train() method, determine_best_model() method will be invoked
            based on pickled models whose hyperparameters were tuned in grid_search.py.
        :param bool override: whether to ignore existing save features data files and rewrite them.
            Set to True if changing features data is desired.`
        """
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        self.model = model

        self.data_loaded = False
        self.model_trained = False
        self.override = override

    def load_data(self):
        """
        Loads features data split into train, val, and test done in data_preparation.py,
        as well as class labels.
        """
        self.X_train, self.X_val, self.X_test,\
            self.y_train, self.y_val, self.y_test, self.labels = prepare_data(self.override, mean=True, type='ml')
        self.data_loaded = True
        print("Data loaded successfully.")

    def grid_search(self, params):
        """
        Performs GridSearchCV on self.model with provided parameters.

        :param dict params: dict with parameters names (str) as keys
        and parameter settings (list) to try as values
        :return: estimator which gave highest accuracy on holdout data,
                 param setting that gave best results on holdout data,
                 mean CV score of best estimator
        :rtype: sklearn model, dict, float
        """
        print(f"Performing GridSearchCV on {self.model}...")
        n_cpu = os.cpu_count()
        grid = GridSearchCV(self.model, params, scoring='accuracy',
                            cv=5,
                            verbose=1,
                            n_jobs=n_cpu-1)
        grid.fit(self.X_train, self.y_train)
        return grid.best_estimator_, grid.best_params_, grid.best_score_

    @staticmethod
    def get_best_classifiers():
        """
        Loads classifiers pickled in grid folder. Classifiers obtained by running grid_search.py.

        :return: list of lists of best estimator, best param setting, best cv score
        :rtype: list
        """
        with open("grid/best_classifiers.pickle", 'rb') as f:
            return pickle.load(f)

    def train_score(self):
        return accuracy_score(y_true=self.y_train,
                              y_pred=self.model.predict(self.X_train))

    def val_score(self):
        return accuracy_score(y_true=self.y_val,
                              y_pred=self.model.predict(self.X_val))

    def determine_best_model(self):
        """
        Loads classifiers (whose hyperparameters were tuned) through get_best_classifiers method,
        then finds the best one with the highest val accuracy, and assigns it to self.model.
        """
        if not self.data_loaded:
            self.load_data()

        # Determine which classifier model is best based on accuracy
        print("Determining best model...")
        estimators = self.get_best_classifiers()
        result = []
        for estimator, params, cv_score in estimators:
            rec = EmotionRecognizer(model=estimator, override=False)
            rec.X_train = self.X_train
            rec.X_val = self.X_val
            rec.X_test = self.X_test
            rec.y_train = self.y_train
            rec.y_val = self.y_val
            rec.y_test = self.y_test
            rec.data_loaded = True

            rec.train()
            acc = rec.val_score()

            result.append((rec.model, acc))

        # Sort by accuracy
        result.sort(key=lambda x: x[1], reverse=True)
        best_estimator, best_acc = result[0]
        print(f"Best model is {best_estimator} with {best_acc:.2f} validation accuracy.")

        self.model = best_estimator
        self.model_trained = True

    def train(self):
        """
        Trains instance attribute model over training data.
        If None was passed as model, method runs 'self.determine_best_model()'.
        """
        if not self.data_loaded:
            self.load_data()

        if not self.model:
            self.determine_best_model()

        if not self.model_trained:
            self.model.fit(self.X_train, self.y_train)
            self.model_trained = True

    def predict(self):
        """
        Evaluates model on test data and prints test accuracy.
        """
        if not self.model_trained:
            self.train()

        print("Predicting on test data...")
        print(f"Accuracy of our model on test data: {self.model.evaluate(self.X_test, self.y_test)[1] * 100}%")

    def plot_confusion_matrix(self, data):
        """
        Prints classification report and plots confusion matrix
        of model based on either validation or test data.

        :param str data: 'val' or 'test'
        """
        if not self.model_trained:
            self.train()

        if data == 'val':
            y_true = self.y_val
            y_pred = self.model.predict(self.X_val)
        else:
            y_true = self.y_test
            y_pred = self.model.predict(self.X_test)

        print(f"Classification Report of {self.model.__class__.__name__} Model on {data.capitalize()} Data")
        print(classification_report(y_true, y_pred, target_names=self.labels))

        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=self.labels,
                                                cmap='Blues', normalize='true')
        plt.title(f"Confusion Matrix of {self.model.__class__.__name__} Model on {data.capitalize()} Data")
        plt.show()

def plot_model_comparisons():
    if os.path.exists("./ml_models_accs.pickle"):
        with open('ml_models_accs.pickle', 'rb') as f:
            result = pickle.load(f)

    else:
        # Load data once
        X_train, X_val, X_test, y_train, y_val, y_test, labels = prepare_data(override=False, mean=True, type='ml')

        # Get model + train acc + val acc data
        estimators = EmotionRecognizer.get_best_classifiers()
        result = []  # (model name, train acc, val acc)
        for estimator, params, cv_score in estimators:
            rec = EmotionRecognizer(model=estimator, override=False)
            rec.X_train = X_train
            rec.X_val = X_val
            rec.X_test = X_test
            rec.y_train = y_train
            rec.y_val = y_val
            rec.y_test = y_test
            rec.data_loaded = True

            print(f"{rec.model.__class__.__name__} training...")
            rec.train()
            print("Done.")
            train_acc = rec.train_score()
            val_acc = rec.val_score()

            result.append((rec.model.__class__.__name__, train_acc, val_acc))

        # Save
        with open('ml_models_accs.pickle', 'wb') as f:
            pickle.dump(result, f)

    # Plot model + accs data
    labels, ys_train, ys_val = zip(*result)  # separate model names and accs
    xs = np.arange(len(labels))  # to make sure it's the same consecutive format as in result
    labels = [text.split('Classifier', 1)[0] for text in labels]

    fig, ax = plt.subplots(figsize=(10,5))
    bar_train = ax.bar(xs, ys_train, label='training', align='edge', width=0.4)
    ax.bar_label(bar_train, fmt='{:.2f}')
    bar_val = ax.bar(xs+0.4, ys_val, label='validation', align='edge', width=0.4)
    ax.bar_label(bar_val, fmt='{:.2f}')

    ax.set_xticks(xs + 0.8 / 2, labels)
    ax.set_title("Accuracies of Various ML Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # plt.bar(xs, ys_train)
    # plt.xticks(xs, labels, rotation=45)
    # plt.title("Training Accuracy of Various ML Models")
    # plt.legend()
    # plt.show()
    #
    # plt.bar(xs, ys_val)
    # plt.xticks(xs, labels, rotation=45)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.title("Validation Accuracy of Various ML Models")
    # plt.legend()
    # plt.show()
