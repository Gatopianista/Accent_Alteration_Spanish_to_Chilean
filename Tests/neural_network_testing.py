from codigo.Neural_Network import modelsetup
import tensorflow as tf
import numpy as np

data_path = 'T:/Usuarios/javiu/Documentos/Máster/TFM/Accent_Alteration_Spanish_to_Chilean/Dataset/True_dataset/data_20/'

train_percent = 90
epochs = 5000
learning_rate = 0.001
batch_size = 16
display_step = 1

n_hidden_1 = 100
n_hidden_2 = 100
n_input = n_output = n_mfccs = 20
loss_fcn = 'mean_squared_error'
metrics = 'accuracy'

data_es = np.load(data_path + 'normalizedDataES.npy')
data_ch = np.load(data_path + 'normalizedDataCH.npy')

x_train, y_train, x_test, y_test = modelsetup.tensor_corpus(data_es, data_ch, n_mfccs, train_percent)

model = modelsetup.neural_network(n_input, n_hidden_1, n_hidden_2, n_output, learning_rate, loss_fcn, metrics)

# model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
model.fit(x_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_size, validation_data=(x_test, y_test))
