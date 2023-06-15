import utils

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
# import librosa.display
# import librosa.feature
# from IPython.display import Audio

from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.utils import to_categorical


class AudioPreparation:
    """
    A class that prepares the data to be fed into the models for training and testing.
    It loads audio files, splits train and test data, balances data, augments data, and extracts features from data.
    """

    def __init__(self, override=False, mean=False, type='ml'):
        """
        :param bool override: whether to ignore existing save features data files and rewrite them.
            Set to True if changing features data is desired.
        :param bool mean: whether to take the mean of MFCC sequence.
            Setting this to True will result in features having a length of 40 (recommended for ML),
            instead of a sequence of 40 being appended at every moment of time (recommended for DL).
        :param str type: whether we're preparing data for 'deep' or 'ml'.
        """
        self.crema = "./data/Crema"
        self.ravdess = "./data/Ravdess/audio_speech_actors_01-24"
        self.savee = "./data/Savee"
        self.tess = "./data/Tess"

        self.df = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.labels = None

        self.override = override
        self.mean = mean
        self.type = type


    def load_data(self):
        """
        Takes path and emotion from each of the 4 datasets and combines them into one dataset.
        """
        # crema
        crema_df = []
        for file in os.listdir(self.crema):
            info = file.split('.')[0].split('_')  # example: 1001_DFA_ANG_XX.wav
            emotion = info[2]
            crema_df.append([emotion, os.path.join(self.crema, file)])

        crema_df = pd.DataFrame(crema_df).rename(columns={0: 'emotion', 1: 'path'})
        crema_emotion_dict = {'SAD': 'sad',
                              'ANG': 'angry',
                              'DIS': 'disgust',
                              'FEA': 'fear',
                              'HAP': 'happy',
                              'NEU': 'neutral'}
        crema_df.emotion.replace(crema_emotion_dict, inplace=True)

        # ravdess
        ravdess_df = []
        for dir_ in os.listdir(self.ravdess):
            actor = os.listdir(os.path.join(self.ravdess, dir_))  # actor folders in Ravdess
            for file in actor:
                info = file.split('.')[0].split('-')
                emotion = int(info[2])
                ravdess_df.append([emotion, os.path.join(self.ravdess, dir_, file)])

        ravdess_df = pd.DataFrame(ravdess_df).rename(columns={0: 'emotion', 1: 'path'})
        ravdess_df.emotion.replace(
            {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
            inplace=True)

        # savee
        savee_df = []
        for file in os.listdir(self.savee):
            info = file.split('.')[0].split('_')  # example: DC_a05.wav
            emotion = info[1][0]
            path = os.path.join(self.savee, file)

            if emotion == 'a':
                savee_df.append(['angry', path])
            elif emotion == 'd':
                savee_df.append(['disgust', path])
            elif emotion == 'f':
                savee_df.append(['fear', path])
            elif emotion == 'h':
                savee_df.append(['happy', path])
            elif emotion == 'n':
                savee_df.append(['neutral', path])
            elif emotion == 'sa':
                savee_df.append(['sad', path])
            else:
                savee_df.append(['surprise', path])

        savee_df = pd.DataFrame(savee_df).rename(columns={0: 'emotion', 1: 'path'})

        # tess
        tess_df = []
        for dir_ in os.listdir(self.tess):
            for file in os.listdir(os.path.join(self.tess, dir_)):
                info = file.split('.')[0].split('_')  # example: OAF_back_angry.wav
                emotion = info[2]
                path = os.path.join(self.tess, dir_, file)

                if emotion == 'ps':
                    tess_df.append(['surprise', path])
                else:
                    tess_df.append([emotion, path])

        tess_df = pd.DataFrame(tess_df).rename(columns={0: 'emotion', 1: 'path'})

        # combine all datasets
        df = pd.concat([crema_df, ravdess_df, savee_df, tess_df], ignore_index=True)

        # set data df to be object attribute
        self.df = df

    def split_data(self):
        """
        Splits data into 60/20/20 train/validation/test sets.
        """
        X = self.df.path
        y = self.df.emotion
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                random_state=42,
                                                                                shuffle=True)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.25,
                                                                              random_state=42,
                                                                              shuffle=True)

    def balance_data(self):
        # Imbalanced data - change calm labels to neutral (little nuance between calm and neutral audios)
        for data in [self.y_train, self.y_val, self.y_test]:
            for i in range(len(data)):
                if data.iloc[i] == 'calm':
                    data.iloc[i] = 'neutral'

        # Imbalanced data - oversample surprise in train only
        train = pd.concat([self.X_train, self.y_train], axis=1)  # easier to work with resampling from one data
        minority = train[train.emotion == 'surprise']
        minority_upsampled = resample(minority, replace=True, n_samples=1400,
                                      random_state=42)  # oversample with replacement
        train.drop(minority.index, inplace=True)  # delete surprise from train
        train = pd.concat([train, minority_upsampled])  # re add surprise with oversampling
        train = shuffle(train, random_state=42)
        self.X_train, self.y_train = train.path, train.emotion

    # TODO: move to utils
    @staticmethod
    def augment_data(data, sr):
        """
        This function injects noise to, shifts the time of, and changes the speed and pitch of data.
        """

        def noise(threshold=0.075):
            """Add random noise to audio sample given some threshold."""
            nonlocal data
            rate = np.random.random() * threshold
            noise_amp = rate * np.random.uniform() * np.amax(data)
            data = data + noise_amp * np.random.normal(size=data.shape[0])
            return data

        def shift(rate=1000):
            """Shift data with some rate."""
            # nonlocal data
            shift_range = int(np.random.uniform(low=-5, high=5) * rate)
            return np.roll(data, shift_range)  # np.roll used for generating time shifting

        def stretch(rate=0.8):  # rate<1 = slowed down, first pass don't want exaggerated stretches so no random
            """Stretch data with some rate."""
            # nonlocal data
            return librosa.effects.time_stretch(data, rate=rate)

        def pitch(pitch_rate=0.7):
            """Add random pitch to audio."""
            # nonlocal data, sr
            pitch_rate = np.random.random() * pitch_rate
            return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_rate)

        return [data, noise(), shift(), stretch(), pitch()]
        # return [data]

    # TODO: move to utils
    # @staticmethod
    def extract_features(self, data, sr):
        # return np.hstack(
        #                     (
        #                         utils.mfcc(data, sr),
        #                         utils.mel_spc(data, sr)
        #                     )
        # )
        return utils.mfcc(data, sr, self.mean)

    # TODO: simplify and/or move to utils
    def write_csv(self, data):
        if data == 'train':
            train = pd.concat([self.X_train, self.y_train.rename('emotion')], axis=1)
            train.to_csv(f"./data_cleaned_features/{self.type}/train.csv", index=False)
            print("Train features data saved to csv file.")

        elif data == 'val':
            val = pd.concat([self.X_val, self.y_val.rename('emotion')], axis=1)
            val.to_csv(f"./data_cleaned_features/{self.type}/val.csv", index=False)
            print("Val features data saved to csv file.")

        elif data == 'test':
            test = pd.concat([self.X_test, self.y_test.rename('emotion')], axis=1)
            test.to_csv(f"./data_cleaned_features/{self.type}/test.csv", index=False)
            print("Test features data saved to csv file.")

    def prepare_train_data(self):
        """
        Apply data augmentation and feature extraction to training data.
        """
        # If featurized already, why redo it
        if os.path.exists(f"./data_cleaned_features/{self.type}/train.csv") and not self.override:
            print("Reading in train data...")
            train = pd.read_csv(f"./data_cleaned_features/{self.type}/train.csv")
            self.X_train = pd.DataFrame(train[train.columns[train.columns != 'emotion']])
            self.y_train = pd.Series(train.emotion)
            print(f"Train features shape: {self.X_train.shape}")
            return

        X_train_feats = []
        y_train_feats = []
        for path, emotion, i in zip(self.X_train, self.y_train, range(len(self.X_train))):
            data, sampling_rate = librosa.load(path, duration=3, offset=0.5)
            num_data = self.augment_data(data, sampling_rate)  # returns num data + augmented versions => 5 copies
            for x in num_data:
                X_train_feats.append(self.extract_features(x, sampling_rate))
                y_train_feats.append(emotion)  # => same emotion gets appended 5x

            if i % 100 == 0:
                if i == 0:
                    print(f"Augmenting and extracting features from training data...")
                else:
                    print(f"{i} training features samples have been processed...")
        print("Training features data finished.")

        # Update training data from path/emo to features/emo
        self.X_train = pd.DataFrame(X_train_feats)
        self.y_train = pd.Series(y_train_feats)

        # Fill null values
        self.X_train = self.X_train.fillna(0.0)

        # Save
        self.write_csv('train')
        print(f"Train features shape: {self.X_train.shape}")

    def prepare_val_data(self):
        """
        Apply feature extraction to val data.
        """
        # If featurized already, why redo it
        if os.path.exists(f"./data_cleaned_features/{self.type}/val.csv") and not self.override:
            print("Reading in val data...")
            val = pd.read_csv(f"./data_cleaned_features/{self.type}/val.csv")
            self.X_val = pd.DataFrame(val[val.columns[val.columns != 'emotion']])
            self.y_val = pd.Series(val.emotion)
            print(f"Val features shape: {self.X_val.shape}")
            return

        X_val_feats = []
        y_val_feats = []
        for path, emotion, i in zip(self.X_val, self.y_val, range(len(self.X_val))):
            data, sampling_rate = librosa.load(path, duration=3, offset=0.5)
            X_val_feats.append(self.extract_features(data, sampling_rate))
            y_val_feats.append(emotion)

            if i % 100 == 0:
                if i == 0:
                    print(f"Extracting features from val data...")
                else:
                    print(f"{i} val features samples have been processed...")
        print("Val features data finished.")

        # Update test data from path/emo to features/emo
        self.X_val = pd.DataFrame(X_val_feats)
        self.y_val = pd.Series(y_val_feats)

        # Fill null values
        self.X_val = self.X_val.fillna(0.0)

        # Save
        self.write_csv('val')
        print(f"Val features shape: {self.X_val.shape}")

    def prepare_test_data(self):
        """
        Apply feature extraction to test data.
        """
        # If featurized already, why redo it
        if os.path.exists(f"./data_cleaned_features/{self.type}/test.csv") and not self.override:
            print("Reading in test data...")
            test = pd.read_csv(f"./data_cleaned_features/{self.type}/test.csv")
            self.X_test = pd.DataFrame(test[test.columns[test.columns != 'emotion']])
            self.y_test = pd.Series(test.emotion)
            print(f"Test features shape: {self.X_test.shape}")
            return

        X_test_feats = []
        y_test_feats = []
        for path, emotion, i in zip(self.X_test, self.y_test, range(len(self.X_test))):
            data, sampling_rate = librosa.load(path, duration=3, offset=0.5)
            X_test_feats.append(self.extract_features(data, sampling_rate))
            y_test_feats.append(emotion)

            if i % 100 == 0:
                if i == 0:
                    print(f"Extracting features from test data...")
                else:
                    print(f"{i} test features samples have been processed...")
        print("Test features data finished.")

        # Update test data from path/emo to features/emo
        self.X_test = pd.DataFrame(X_test_feats)
        self.y_test = pd.Series(y_test_feats)

        # Fill null values
        self.X_test = self.X_test.fillna(0.0)

        # Save
        self.write_csv('test')
        print(f"Test features shape: {self.X_test.shape}")

    def encode_labels(self):
        lb = LabelEncoder()
        self.y_train = lb.fit_transform(self.y_train)
        if self.type == 'deep':
            self.y_train = pd.DataFrame(to_categorical(self.y_train))
        else:
            self.y_train = pd.Series(self.y_train)

        self.y_test = lb.transform(self.y_test)
        if self.type == 'deep':
            self.y_test = pd.DataFrame(to_categorical(self.y_test))
        else:
            self.y_test = pd.Series(self.y_test)

        self.y_val = lb.transform(self.y_val)
        if self.type == 'deep':
            self.y_val = pd.DataFrame(to_categorical(self.y_val))
        else:
            self.y_val = pd.Series(self.y_val)

        self.labels = lb.classes_
        print(f"Classes: {self.labels}")

    def standardize_features(self):
        # Pad val features data with 0s to match number of features in training features data
        df_zeros = pd.DataFrame(index=range(self.X_val.shape[0]),
                                columns=range(self.X_train.shape[1] - self.X_val.shape[1]))
        df_zeros = df_zeros.fillna(0.0)
        df_zeros.columns = [i for i in range(self.X_val.shape[1], self.X_train.shape[1])]

        self.X_val = pd.concat([self.X_val, df_zeros], axis=1)
        self.X_val.columns = self.X_val.columns.astype(str)

        print(f"Val features padded to match train features shape.\nVal features shape: {self.X_val.shape}")

        # Pad test features data with 0s to match number of features in training features data
        df_zeros = pd.DataFrame(index=range(self.X_test.shape[0]),
                                columns=range(self.X_train.shape[1] - self.X_test.shape[1]))
        df_zeros = df_zeros.fillna(0.0)
        df_zeros.columns = [i for i in range(self.X_test.shape[1], self.X_train.shape[1])]

        self.X_test = pd.concat([self.X_test, df_zeros], axis=1)
        self.X_test.columns = self.X_test.columns.astype(str)

        print(f"Test features padded to match train features shape.\nTest features shape: {self.X_test.shape}")

        # Standardize data
        ss = StandardScaler()
        self.X_train = pd.DataFrame(ss.fit_transform(self.X_train))  # turns input to array
        self.X_val = pd.DataFrame(ss.transform(self.X_val))
        self.X_test = pd.DataFrame(ss.transform(self.X_test))


def prepare_data(override, mean, type):
    """
    :param bool override: whether to ignore existing save features data files and rewrite them.
        Set to True if changing features data is desired.
    :param bool mean: whether to take the mean of MFCC sequence.
        Setting this to True will result in features having a length of 40 (recommended for ML),
        instead of a sequence of 40 being appended at every moment of time (recommended for DL).
    :param str type: whether we're preparing data for 'deep' or 'ml'.

    :return: features of each training, validation, and test sets,
        their labels, and the corresponding emotions of the labels.
    """
    # Instantiate the class
    audio_preparation = AudioPreparation(override, mean, type)

    # Preprocess data. This is still path/emo data
    audio_preparation.load_data()
    audio_preparation.split_data()
    audio_preparation.balance_data()

    # Augment data on training only and extract features on training, validation, and test
    audio_preparation.prepare_train_data()
    audio_preparation.prepare_val_data()
    audio_preparation.prepare_test_data()

    # TODO: When saving train and test datasets, save with path as well

    # Final preparation of features and target
    audio_preparation.encode_labels()
    audio_preparation.standardize_features()

    return audio_preparation.X_train, audio_preparation.X_val, audio_preparation.X_test,\
           audio_preparation.y_train, audio_preparation.y_val, audio_preparation.y_test,\
           audio_preparation.labels
