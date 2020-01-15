import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import time


# from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
# from tensorflow.keras.callbacks import Tensorboard, ModelCheckpoint
# import numpy as np
# import random

def tensor_corpus(input, output, n_mfccs, test_prcnt):
    """
    Función que prepara los datos para ser configurados en tensor

    :param input: Aquí se introducen los valores equivalentes a la entrada de la red neuronal
    :param output: Aquí se introducen los valores de los target correspondientes a las entradas
    :param test_prcnt: Este es el porcentaje (valor entero 0~100) que define la cantidad de muestras de entrenamiento
    :return: inputs de entrenamiento, targets de entremaniento, inputs de test, targets de test
    """
    # Creación del tensor
    all_data = tf.constant([input, output])
    # Conteo de datos de entrenamiento/test
    n_samples = all_data.shape[1]
    n_train_samples = round(n_samples * test_prcnt / 100)
    n_test_samples = n_samples - n_train_samples
    train = tf.slice(all_data, [0, 0, 0], [2, n_train_samples, n_mfccs])
    test = tf.slice(all_data, [0, n_train_samples, 0], [2, n_test_samples, n_mfccs])

    x_train, y_train = train[0], train[1]
    x_test, y_test = test[0], test[1]

    return x_train, y_train, x_test, y_test


def neural_network(n_input, n_hidden_1, n_hidden_2, n_output, lr, loss_fcn, metrics):
    """
    Función que define la red neuronal secuencial

    :param n_input: Dimensión de la capa de entrada (nº de mfccs que se introducirán en la red)
    :param n_hidden_1: Dimensión de la capa oculta 1
    :param n_hidden_2: Dimensión de la capa oculta 2
    :param n_output: Dimensión de la capa de salida (nº de mfccs que devolverá la red)
    :param lr: learning rate (velocidad a la que el optimizador funcionará)
    :param loss_fcn: ('String') Función de pérdidas
    :param metrics: ('String') Función de exactitud
    :return: model: Red neuronal resultante
    """

    model = Sequential([
        # Dropout(0.2),
        Dense(n_hidden_1, activation='tanh', input_dim=n_input),
        # Dropout(0.2),
        Dense(n_hidden_2, activation='tanh'),
        # Dropout(0.2),
        Dense(n_output, activation='relu'),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, decay=1e-3)

    model.compile(loss=loss_fcn, optimizer=optimizer, metrics=[metrics])

    return model


def trainer(model, nf, epochs, x_train, y_train, x_test, y_test):
    dt = datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H-%M-%S')
    name = f"model-{nf}f-{dt}"
    dirname = f"FF_Neural_Network/model-{nf}f-{dt}"
    os.mkdir(dirname)
    os.mkdir(dirname + "/checkpoints")

    tensorboard = TensorBoard(log_dir=f"logs\{name}")
    filepath = dirname + "/checkpoints/{epoch:02d}"
    checkpoint = ModelCheckpoint("{}.model".format(filepath, monitor='val_acc', verbose=1,
                                                   save_best_only=True, mode='max'))
    hist = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),
                     callbacks=[tensorboard, checkpoint])
    return hist, name
