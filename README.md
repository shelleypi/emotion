# Speech Emotion Recognition
Speech emotion recognition (SER) is the task of recognizing the emotional aspects of speech irrespective of its semantic contents. While humans can efficiently perform this task, a computer's ability to conduct it is still an ongoing research. Adding emotions to machines has been recognized as a critical factor in making machines appear and act in a human-like manner.

In this project, we will attempt to recognize human emotion from speech, since human voice often reflects the underlying emotion through tone and pitch.

Relevance:
- Emotion recognition, which is part of speech recognition, is gaining more popularity within the last decade and the need for it increases enormously. Although there are methods to recognize emotion through machine learning techniques, this project also attempts to do it through the use of deep learning.
- SER can be used in <b>customer service</b> calls as performance parameter for conversational analysis to detect customer satisfaction, thus helping companies improve their service.
- SER also has applications in the <b>healthcare</b>, namely for aiding in the early detection of Alzheimer's Disease and dementia, or simply to discern a patient's feelings in the healing process through telehealth.

## Results
Baseline model of CNN input + 1 hidden layer achieved 58% validation accuracy. The final CNN-LSTM model achieved 71% validation accuracy--a 13% increase. Test accuracy is 72%.

## Technologies
This project uses the following versions:
- Python 3.10
- NumPy 1.23.2
- pandas 2.0.0
- Matplotlib 3.7.1
- Librosa 0.10.0.post2
- scikit-learn 1.2.2
- TensorFlow 2.9.0
- Keras 2.9.0

## Datasets
The following datasets can be located in the `data` folder.
- Crowd-sourced Emotional Mutimodal Actors Dataset (Crema)
- Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)
- Surrey Audio-Visual Expressed Emotion (Savee)
- Toronto Emotional Speech Set (Tess)

This project uses the following 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise.

## Setup
By default, the `main.py` script runs the deep learning model, but you can un/comment to run the machine learning model.
```
python main.py
```

You can also execute
```
python grid_search.py
```
to determine the best parameters for various machine learning classifiers. The pickle file in the `grid` folder contains the results.

## Structure
- `data_preparation.py`: all data preparation happens in this script. The `prepare_data` function handles the whole script and returns the features and labels of each training, validation, and test sets, as well as the corresponding emotions of the encoded labels.
- `emotion_recognition.py`: this script consists of the machine learning `EmotionRecognizer` class.
- `emotion_recognition_deep.py`: deep learning version of `EmotionRecognizer` class.
- `hyperparams.py`: contains a dictionary of various machine learning estimators and their hyperparameters.
- `grid_search.py`: you can run this script to get the best parameters for each estimator listed in `hyperparams.py`.
- `utils.py`: consists mostly of functions for feature extraction of audios. By default, this project uses `mfcc` only, but there are other feature extraction functions in this script that you can also use alongside or other than `mfcc`.
- `main.py`: start of execution. 
- `ml_models_accs.pickle`: contains training and validation accuracies of the various ML classifiers.
- Folders:
  - `data`: contains the folders of the 4 datasets.
  - `data_cleaned_features`: contains the train/val/test CSVs for ML and DL
  - `grid`: contains the pickle file of the best parameters for various ML classifiers
  - `images`: contains plots for ML and DL
  - `models`: contains the saved models for DL (baseline and final model) (If you set override to True in `main.py`, these saved models will get overwritten.)

## Additional Files
The following links contain the `data_cleaned_features` and `grid` folders:
- [data_cleaned_features](https://drive.google.com/drive/folders/1JbNNcteaNpVoUtXEwzvhIm2tzpIHuPqS?usp=sharing)
- [grid](https://drive.google.com/drive/folders/19AOdh0pEsTkbjy6un02mrkiul30OObk4?usp=sharing)
