from codigo.Audio_Process import audioprocessor as process
import numpy as np
import sys


def sorter(dirES, dirCH):
    # Esta función recoge los archivos de audio, los deja ordenados en dos vectores y los devuelve
    AudioArrayES, sizeES = process.files(dirES)
    AudioArrayCH, sizeCH = process.files(dirCH)
    if sizeES != sizeCH:
        sys.exit("Number of samples and target samples are different!")
    ArraySize = sizeES
    return AudioArrayES, AudioArrayCH, ArraySize


def audioMutilator(AudioArrayES, AudioArrayCH, nSplits):
    # Esta funcion silencia, iguala duración y divide todos los audios, dejando todos los trozos de todos los audios en dos arrays
    sizeArray = len(AudioArrayES)
    mutArrayES = []
    mutArrayCH = []

    for i in range(sizeArray):
        # Obtención de propiedades: Señal, velocidad, nº de muestras
        signalES, rateES, nframesES = process.waveanalyzer(AudioArrayES[i])
        signalCH, rateCH, nframesCH = process.waveanalyzer(AudioArrayCH[i])

        # Eliminación de silencio de la señal.
        muffledSignalES = process.muffler(signalES)
        muffledSignalCH = process.muffler(signalCH)

        # Igualación de ambas señales
        eqSignalCH = process.equalizer(muffledSignalES, muffledSignalCH)

        # División de la señal
        splittedSignalES = process.splitter(muffledSignalES, nSplits)
        splittedSignalCH = process.splitter(eqSignalCH, nSplits)

        mutArrayES.append(splittedSignalES)
        mutArrayCH.append(splittedSignalCH)
        print('\r Procesados %d de %d archivos.' % (i+1, sizeArray))

    return mutArrayES, mutArrayCH


def audioCharacterizator(mutArrayES, mutArrayCH, nCoefs):
    rate = 22050
    mfcc_es, mfcc_ch = [], []
    n_audios_ES, n_audios_CH = len(mutArrayES), len(mutArrayCH)
    print('Extrayendo características de las voces en español neutro')
    for i in range(n_audios_ES):
        n_splits = len(mutArrayES[i])
        for m in range(n_splits):
            mfcc = process.MFCCExtractor(mutArrayES[i][m], rate, nCoefs)
            mfcc_es.append(mfcc)
    print('Extrayendo características de las voces en español chileno')
    for i in range(n_audios_CH):
        n_splits = len(mutArrayES[i])
        for m in range(n_splits):
            mfcc = process.MFCCExtractor(mutArrayCH[i][m], rate, nCoefs)
            mfcc_ch.append(mfcc)

    es_coefs_norm, ch_coefs_norm = process.Normalizator(mfcc_es, mfcc_ch)

    return es_coefs_norm, ch_coefs_norm
