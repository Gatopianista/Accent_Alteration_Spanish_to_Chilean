import librosa
import numpy as np
from glob import glob
from scipy import signal
import speechpy

def files(dir):
    # Desde un directorio especifico, recoge todos los audios dentro de el
    # y los incorpora en un array para manejarlos mejor
    files = glob(dir + '/*.wav')
    size = len(files)
    return files, size


def waveanalyzer(file):
    signal, rate = librosa.load(file, mono=True)
    nframes = len(signal)
    return signal, rate, nframes


def muffler(signal):
    # Elimina el silencio de un audio, recortándolo para dejar despejado el habla
    muffledsignal = signal[0]
    for i in range(len(signal)):
        if signal[i] > 0.001 or signal[i] < -0.001:
            muffledsignal = np.append(muffledsignal, signal[i])
    return muffledsignal


def equalizer(signal1, signal2):
    # Enlentece/apresura una onda de sonido tanto como haga falta.
    # Se usará para igualar el tiempo de dos muestras.
    nframes1 = len(signal1)
    eqSignal = signal.resample(signal2, nframes1)
    return eqSignal


def splitter(signal, splits):
    # Divide un audio en varios trozos de igual longitud
    duration = len(signal)
    rest = duration % splits
    ext = np.zeros((splits - rest), dtype=int)
    signal_ext = np.append(signal, ext)
    c = len(signal_ext) / splits
    splittedsignal = signal_ext.reshape(splits, int(c))
    return splittedsignal


def MFCCExtractor(signal, sr, nCoefs):
    # Obtiene tantos MFCCs como le definas de una onda.
    MelCoefs = speechpy.feature.mfcc(signal, sr, frame_length=0.025, frame_stride=0.010, num_cepstral=nCoefs)
    return MelCoefs


def Normalizator(es_coefs, ch_coefs):
    # Normaliza los valores de los MFCCS del audio de entrada y del target. Tras esto las muestras estarían listas para la red
    all_coefs = es_coefs[0]
    for i in range(len(es_coefs) - 1):
        all_coefs = np.concatenate((all_coefs, es_coefs[i + 1]), axis=0)
    for i in range(len(ch_coefs)):
        all_coefs = np.concatenate((all_coefs, ch_coefs[i]), axis=0)
    all_coefs_norm = speechpy.processing.cmvn(all_coefs)
    es_coefs_norm = all_coefs_norm[0:int(len(all_coefs_norm)/2)][:]
    ch_coefs_norm = all_coefs_norm[int(len(all_coefs_norm)/2):][:]

    return es_coefs_norm, ch_coefs_norm
