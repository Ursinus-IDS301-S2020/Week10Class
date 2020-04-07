"""
Programmer: Chris Tralie
Purpose: To develop a system for quick n' dirty genre classification
based on ideas and data from
[1]Tzanetakis, George, and Perry Cook. "Musical genre classification of audio signals." IEEE Transactions on speech and audio processing 10.5 (2002): 293-302.

DISCLAIMER: The notion of "genre" as described in this paper has been highly
contested since the paper was written.  We are merely using this as a simple 
example since the data is available

"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.io as sio
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def get_audio_features(filename, texture_win=43):
    """
    Compute a simple quick-n-dirty set of features based on
    standard deviations/means of standard deviations/means
    of chroma and mfcc features in windows.  Leads to about
    55% accuracy with 5-nearest neighbors.

    Parameters
    ----------
    filename: string
        Path to audio file
    texture_win: int
        The number of analysis windows in each texture window
    """
    import librosa
    y, sr = librosa.load(filename)
    hop_length=512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = chroma.T
    mfcc = mfcc.T
    all_feats = np.concatenate((mfcc, chroma), axis=1)
    all_feats /= np.std(all_feats, 0)[None, :]
    N = all_feats.shape[0]
    d = all_feats.shape[1]
    M = N-texture_win+1

    X = np.zeros((M, d*2))
    for i in range(M):
        x = all_feats[i:i+texture_win, :]
        X[i, 0:d] = np.mean(x, axis=0)
        X[i, d::] = np.std(x, axis=0)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return np.concatenate((means, stds))

def precompute_all_features():
    """
    Precompute all of the features in the Tzanetakis
    dataset 
    """
    X = []
    for genre in GENRES:
        foldername = "GTzanmp3/{}".format(genre)
        for f in glob.glob("{}/*.mp3".format(foldername)):
            print(f)
            X.append(get_audio_features(f))
    X = np.array(X)
    sio.savemat("GenreFeatures.mat", {"X":X})


def classify_KNN():
    """
    Use sklearn's built in KNN classifier with 10-fold cross-validation
    to see how well we can separate genre with these features
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    X = sio.loadmat("GenreFeatures.mat")["X"]
    labels = np.floor(np.arange(1000)/100)
    knn = KNeighborsClassifier(n_neighbors = 5)
    scores = cross_val_score(knn, X, labels, cv=10, scoring='accuracy')
    print(scores, np.mean(scores))

def plot_genres():
    """
    Do PCA on the genres data to visualize it
    """
    X = sio.loadmat("GenreFeatures.mat")["X"]
    X = X/np.sqrt(np.sum(X**1, 1)[:, None])
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)
    plt.figure(figsize=(10, 10))
    for i in range(10):
        y = Y[i*100:(i+1)*100, :]
        plt.scatter(y[:, 0], y[:, 1], marker=i)
    plt.legend(GENRES)

def classify_song(filename, n_neighbors=20):
    """
    Guess the genre of a song with nearest neighbors
    """
    X = sio.loadmat("GenreFeatures.mat")["X"]
    q = get_audio_features(filename)
    qflat = q.flatten()[None, :]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X)
    distances, neighbors = nbrs.kneighbors(qflat)
    distances = distances.flatten()
    neighbors = neighbors.flatten()

    neighbors = np.floor(neighbors/100)
    for n in neighbors:
        n = int(n)
        print(GENRES[n])
    mode = scipy.stats.mode(neighbors)[0]
    mode = int(mode)
    print("Guess: ", GENRES[mode])

## TODO: Classify your own song
classify_song("del.mp3")