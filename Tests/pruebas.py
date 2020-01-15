from Audio_Process import audio_feature_processing
import numpy as np
from glob import glob
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from Audio_Process import audioprocessor as process
import librosa

print('LISTO PARA EMPEZAR')
path = 'T:/Usuarios/javiu/Documentos/Máster/TFM/Dataset/pruebas'
pathFilesES = 'T:/Usuarios/javiu/Documentos/Máster/TFM/Dataset/es_spanish_male'
pathFilesCH = 'T:/Usuarios/javiu/Documentos/Máster/TFM/Dataset/es_chilean_male'

filesES, sizeFilesES = process.files(pathFilesES)
print('Número de archivos en español: ' + str(sizeFilesES))
filesCH, sizeFilesCH = process.files(pathFilesCH)
print('Número de archivos en chileno: ' + str(sizeFilesCH))

mutArrayES = []
mutArrayCH = []


"""
for i in range(int(sizeFilesES/5)):
    signalES, rateES, nframesES = process.waveanalyzer(filesES[i])
    signalCH, rateCH, nframesCH = process.waveanalyzer(filesCH[i])

    muffledSignalES = process.muffler(signalES)
    muffledSignalCH = process.muffler(signalCH)

    eqsignalCH = process.equalizer(muffledSignalES, muffledSignalCH)

    splittedsignalES = process.splitter(muffledSignalES, 10)
    splittedsignalCH = process.splitter(eqsignalCH, 10)

    mutArrayES.insert(i, muffledSignalES)
    mutArrayCH.insert(i, eqsignalCH)


    print(i)
"""

"""
# files = glob(path + '/*.wav')
files = process.files(path)
signal, sr, nframes = process.waveanalyzer('T:/Usuarios/javiu/Documentos/Máster/TFM/Dataset/es_chilean_male/clm_02121_02077117829.wav')
muffledsignal = process.muffler(signal)
equalizedsignal = process.equalizer(muffledsignal, muffledsignal)
splittedsignal = process.splitter(equalizedsignal, splits=100)
for i in range(len(splittedsignal)):
    librosa.output.write_wav('result' + str(i) + '.wav', splittedsignal[i], sr)
"""

"""
plt.figure(1)
plt.plot(signal2)
plt.figure(2)
plt.plot(signal)
plt.show()

mfccs_librosa = audio.extract_features_librosa(files, 20)
mfccs_one_file = audio.mfcc_extraction(files[0])
mfccs_speechpy = audio.extract_features_speechpy(files,20)
"""
