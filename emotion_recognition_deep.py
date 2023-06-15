from emotion_recognition import EmotionRecognizer
from data_preparation import prepare_data

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import tensorflow as tf
# from tensorflow.compat.v1.keras.backend import set_session
from keras import backend as k
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPool1D, Flatten, Dense, BatchNormalization, Dropout
from keras.layers import LSTM, ELU, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback

class DeepEmotionRecognizer(EmotionRecognizer):
    def __init__(self, override=False, **kwargs):
        super().__init__()

        # CNN layers
        self.cnn_only = kwargs.get('cnn_only', True)

        self.n_cnn_layers = kwargs.get('n_cnn_layers', 3)
        self.cnn_units = kwargs.get('cnn_units', 64)  # starting units, will double after every layer

        # LSTM layers
        if not self.cnn_only:  # CNN + LSTM
            self.n_rnn_layers = kwargs.get('n_rnn_layers', 2)
            self.rnn_units = kwargs.get('rnn_units', 512)

        # Dense layers
        self.n_dense_layers = kwargs.get('n_dense_layers', 3)
        self.dense_units = kwargs.get('dense_units', 128)  # starting units, will half after every layer

        self.dropout = kwargs.get('dropout', 0.3)

        self.output_dim = len(self.labels)

        self.opt = kwargs.get('opt', 'rmsprop')
        self.loss = kwargs.get('loss', 'categorical_crossentropy')

        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 64)

        self.model = None
        self.history = None

        self.data_loaded = False
        self.model_created = False
        self.model_trained = False
        self.hist_from_file = False
        self.override = override

    def load_data(self):
        """
        Loads the train/test data processed in data_preparation.py,
        reshapes them based on the NN,
        then splits the training data into train/validation.
        """
        # super().load_data(self.override)
        self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test, self.labels = prepare_data(self.override, mean=False, type='deep')

        if self.cnn_only:
            # Reshape for cnn
            self.X_train = np.expand_dims(self.X_train, axis=2)
            self.X_val = np.expand_dims(self.X_val, axis=2)
            self.X_test = np.expand_dims(self.X_test, axis=2)

        else:
            # Reshape for rnn
            # Reshape based on [batches, timesteps, features]
            # batches = n samples
            # timesteps = n features / 40 since we're feeding 40 features at a time (n_mfcc set to 40)
            # features = 40 since feeding data 40 at a time (n_mfcc set to 40)
            self.X_train = self.X_train.values.reshape((self.X_train.shape[0], int(self.X_train.shape[1] / 40), 40))
            self.X_val = self.X_val.values.reshape((self.X_val.shape[0], int(self.X_val.shape[1] / 40), 40))
            self.X_test = self.X_test.values.reshape((self.X_test.shape[0], int(self.X_test.shape[1] / 40), 40))

        self.data_loaded = True

    def create_model(self):
        """
        Creates and compiles a model based on instance attributes.
        """
        if not self.data_loaded:
            self.load_data()

        self.model = Sequential()

        # Input shape
        if self.cnn_only:
            input_shape = (int(self.X_train.shape[1]), 1)
        else:
            input_shape = (int(self.X_train.shape[1]), 40)

        # CNN layers
        for i in range(self.n_cnn_layers):
            if i == 0:
                self.model.add(Conv1D(self.cnn_units, kernel_size=5, padding='same', activation=None,
                                      input_shape=input_shape))
            else:
                self.cnn_units *= 2
                self.model.add(Conv1D(self.cnn_units, kernel_size=5, padding='same', activation=None))
            self.model.add(ELU())  # merger between good features of ReLU and LeakyReLU
            self.model.add(BatchNormalization())
            self.model.add(MaxPool1D(pool_size=5, padding='same'))
            self.model.add(Dropout(self.dropout))

        # LSTM layers
        if not self.cnn_only:
            for i in range(self.n_rnn_layers):
                self.model.add(LSTM(self.rnn_units, return_sequences=True))
                self.model.add(Dropout(self.dropout))

        self.model.add(Flatten())
        self.model.add(Dropout(self.dropout))

        # Dense layers
        for i in range(self.n_dense_layers):
            self.model.add(Dense(self.dense_units, activation=None))
            self.model.add(LeakyReLU())
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout))
            self.dense_units /= 2
        self.model.add(Dense(self.output_dim, activation='softmax'))

        self.model.compile(optimizer=self.opt, loss=self.loss, metrics=['accuracy'],
                           run_eagerly=True
                           )

        print(self.model.summary())
        self.model_created = True

    class ClearMemory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            k.clear_session()

    def train(self):
        """
        Trains model on training data over 'self.epochs' epochs and 'self.batch_size' batch size,
        and saves the training history, as well as the model with the best performance based on validation accuracy.

        If a saved model exists and 'self.override' is False,
        said model will be loaded along with the training history.
        """
        if not self.model_created:
            self.create_model()

        model_path = f"./models/model_{'cnn' if self.cnn_only else 'with_rnn'}.h5"
        hist_path = f"./models/history_{'cnn' if self.cnn_only else 'with_rnn'}.csv"

        if os.path.exists(model_path) and not self.override:
            self.model = load_model(model_path,
                                    compile=False
                                    )
            self.model.compile(optimizer=self.opt, loss=self.loss, metrics=['accuracy'], run_eagerly=True)
            self.model_trained = True
            print("Model loaded from save file.")
            self.history = pd.read_csv(hist_path)
            self.hist_from_file = True
            return

        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                      factor=0.3,
                                      patience=3,
                                      threshold=0.001,  # min improvement; if below, reduce lr
                                      # cooldown=3,
                                      verbose=1,
                                      min_lr=0.00001)
        stop_early = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0.001,
                                   patience=10,
                                   verbose=1,
                                   restore_best_weights=True)
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1)

        # Restrict amount of memory consumption in tf
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.visible_device_list = "0"
        config.gpu_options.allow_growth = True
        k.set_session(tf.compat.v1.Session(config=config))

        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      workers=0,
                                      callbacks=[reduce_lr,
                                                 # stop_early,
                                                 checkpoint,
                                                 self.ClearMemory()
                                                 ]
                                      )
        self.model_trained = True
        print("Model trained and saved to h5 file.")

        # Save history to csv
        print("Saving model history...")
        history_df = pd.DataFrame(self.history.history)
        with open(hist_path, mode='w') as f:
            history_df.to_csv(f)
        print("Model history saved to csv file.")

    def plot_loss_and_acc(self):
        """
        Shows plots of loss and accuracy of the model
        on the training and validation datasets over training epochs.
        """
        if not self.hist_from_file:
            self.history = self.history.history

        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title(f"{'CNN' if self.cnn_only else 'CNN + LSTM'} Model Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(['train', 'validation'])
        plt.show()

        plt.plot(self.history['accuracy'])
        plt.plot(self.history['val_accuracy'])
        plt.title(f"{'CNN' if self.cnn_only else 'CNN + LSTM'} Model Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(['train', 'validation'])
        plt.show()

    def train_score(self):
        """
        Evaluates model on train data and prints train accuracy.
        """
        if not self.model_trained:
            self.train()

        print("Predicting on train data...")
        acc_score = self.model.evaluate(self.X_train, self.y_train, batch_size=self.batch_size)[1]
        print(f"Accuracy of our model on train data: {acc_score * 100:.2f}%")

    def val_score(self):
        """
        Evaluates model on validation data and prints validation accuracy.
        """
        if not self.model_trained:
            self.train()

        print("Predicting on validation data...")
        acc_score = self.model.evaluate(self.X_val, self.y_val, batch_size=self.batch_size)[1]
        print(f"Accuracy of our model on validation data: {acc_score * 100:.2f}%")

    def predict(self):
        """
        Evaluates model on test data and prints test accuracy.
        """
        if not self.model_trained:
            self.train()

        print("Predicting on test data...")
        acc_score = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)[1]
        print(f"Accuracy of our model on test data: {acc_score * 100:2f}%")

    def plot_confusion_matrix(self, data):
        """
        Prints classification report and plots confusion matrix
        of model based on either validation or test data.

        Deep version needs argmax, since target is one hot encoded.

        :param str data: 'val' or 'test'
        """
        if not self.model_trained:
            self.train()

        if data == 'val':
            y_true = np.argmax(self.y_val, axis=1)
            y_pred = np.argmax(self.model.predict(self.X_val, batch_size=self.batch_size), axis=1)
        else:
            y_true = np.argmax(self.y_test, axis=1)
            y_pred = np.argmax(self.model.predict(self.X_test, batch_size=self.batch_size), axis=1)

        print(f"Classification Report of {'CNN' if self.cnn_only else 'CNN + LSTM'} Model on {data.capitalize()} Data")
        print(classification_report(y_true, y_pred, target_names=self.labels))

        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=self.labels,
                                                cmap='Blues', normalize='true')
        plt.title(f"Confusion Matrix of {'CNN' if self.cnn_only else 'CNN + LSTM'} Model on {data.capitalize()} Data")
        plt.show()
