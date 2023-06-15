import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa.feature

def zcr(data):
    return librosa.feature.zero_crossing_rate(data)[0]

def rms(data):
    return librosa.feature.rms(y=data)[0]

def spectral_centroid(data, sr):
    return librosa.feature.spectral_centroid(y=data, sr=sr)[0]

def ber(data, sr, split_freq=2000):
    spectrogram = librosa.stft(data)

    # Calculate split freq bin
    freq_range = sr / 2
    freq_delta_per_bin = freq_range / spectrogram.shape[0]
    split_freq_bin = int(np.floor(split_freq / freq_delta_per_bin))

    # Move to the power_spectrogram
    power_spec = np.abs(spectrogram) ** 2
    power_spec = power_spec.T

    # Calculate ber for each frame
    ber = []
    for freqs_in_frame in power_spec:
        sum_power_low_freqs = np.sum(freqs_in_frame[:split_freq_bin])
        sum_power_high_freqs = np.sum(freqs_in_frame[split_freq_bin:])
        ber_current_frame = sum_power_low_freqs / sum_power_high_freqs

        ber.append(ber_current_frame)

    return np.array(ber)

def spectral_spread(data, sr):
    return librosa.feature.spectral_bandwidth(y=data, sr=sr)[0]

def mel_spc(data, sr, mean=False):
    if mean:  # for ML, fewer features
        return np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    return np.ravel(librosa.feature.melspectrogram(y=data, sr=sr).T)

def mfcc(data, sr, mean=False):
    if mean:  # for ML, fewer features
        return np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    return np.ravel(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T)   # best to have a fixed n_mfcc for LSTM later

def create_waveplot(data, sr, emo):
    plt.figure(figsize=(10, 3))
    plt.title(f"Waveplot for audio with {emo} emotion")
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def create_spectrogram(data, sr, emo):
    X = librosa.stft(data)  # converts data to short term fourier transform
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title(f"Spectrogram for audio with {emo} emotion")
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
