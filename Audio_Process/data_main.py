from codigo.Audio_Process import datamanager
import numpy as np

# Hiperparámetros
pathFilesES = 'T:/Usuarios/javiu/Documentos/Máster/TFM/Accent_Alteration_Spanish_to_chilean/Dataset/00610_es_ch/spanish'
pathFilesCH = 'T:/Usuarios/javiu/Documentos/Máster/TFM/Accent_Alteration_Spanish_to_chilean/Dataset/00610_es_ch/chilean'
pathData = 'T:/Usuarios/javiu/Documentos/Máster/TFM/Accent_Alteration_Spanish_to_chilean/Dataset/true_dataset/data_15/'
nSplits = 10  # Cortes que se realizan a los audios
nMFCCs = 15  # Número de coeficientes cepstrales extraidos de cada muestra

# Obtención de todos los audios de ambas carpetas en forma de array
AudioArrayES, AudioArrayCH, arraySize = datamanager.sorter(pathFilesES, pathFilesCH)
print('Audios obtenidos')

# Modificación de las ondas para igualarlas lo máximo posible
mutArrayES, mutArrayCH = datamanager.audioMutilator(AudioArrayES, AudioArrayCH, nSplits)
print('Audios modificados')
# Obtención de MFCCs y normalización de las muestras
normalizedES, normalizedCH = datamanager.audioCharacterizator(mutArrayES, mutArrayCH, nMFCCs)
print('Audios normalizados')

np.save(pathData + 'normalizedDataES', normalizedES)
np.save(pathData + 'normalizedDataCH', normalizedCH)
print('Proceso finalizado')
