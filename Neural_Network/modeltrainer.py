from codigo.Neural_Network import modelsetup
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# HIPERPARAMETROS
data_path = 'T:/Usuarios/javiu/Documentos/Máster/TFM/Accent_Alteration_Spanish_to_Chilean/Dataset/True_dataset/data/'

train_percent = 80
epochs = 10
learning_rate = 0.01
batch_size = 32
display_step = 1

n_hidden_1 = 100
n_hidden_2 = 100
n_input = 20
n_output = 20

loss_fcn = 'mean_squared_error'
metrics = 'accuracy'

data_es = np.load(data_path + 'normalizedDataES.npy')
data_ch = np.load(data_path + 'normalizedDataCH.npy')

tf_data = modelsetup.tensor_corpus(data_es, data_ch, train_percent)