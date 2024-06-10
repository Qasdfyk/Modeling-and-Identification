import numpy as np
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def create_data(rzad, data_type, recursive):
    with open('danedynucz33.txt', 'r') as file:
        dane_ucz = []
        for line in file:
            columns = line.split()
            dane_ucz.append([float(columns[0]), float(columns[1])])
    with open('danedynwer33.txt', 'r') as file:
        dane_wer = []
        for line in file:
            columns = line.split()
            dane_wer.append([float(columns[0]), float(columns[1])])
    dane_ucz = np.array(dane_ucz)
    dane_wer = np.array(dane_wer)
    data = dane_ucz if data_type else dane_wer
    u = []
    y = []
    for i in range(rzad-1, len(data[:, 0])):
        row = []
        for j in range(1, rzad):
            row.append(data[i - j, 0])
            row.append(data[i - j, 1])
        u.append(row)
        y.append(data[i, 1])
    u = np.array(u)
    if recursive:
        u = u.reshape((u.shape[0], rzad-1, 2))
    return u, np.array(y)

def neural_network(rzad, neurony, recursive, visualize=True):
    u, y = create_data(rzad, True, recursive)

    model = Sequential()
    if recursive:
        model.add(LSTM(neurony, input_shape=(rzad-1, 2), activation='leaky_relu'))
    else:
        model.add(Dense(neurony, input_dim=2 * (rzad - 1), activation='leaky_relu'))
    model.add(Dense(1, activation='leaky_relu'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(u, y, epochs=100)

    predictions_ucz = model.predict(u)
    u_wer, y_wer = create_data(rzad, False, recursive)
    predictions_wer = model.predict(u_wer)

    predictions_ucz = predictions_ucz.flatten()
    predictions_wer = predictions_wer.flatten()

    mse_ucz = mean_squared_error(y, predictions_ucz)
    mse_wer = mean_squared_error(y_wer, predictions_wer)

    if visualize:
        plt.figure(1)
        plt.plot(range(len(y)), y, label='Zbiór uczący')
        plt.plot(range(len(y)), predictions_ucz, label='Wyjście modelu')
        plt.title(f'Predykcja dla danych uczących, {"z rekurencja"if recursive else "bez rekurencji"}, liczba neuronów: {neurony}')
        plt.xlabel('k')
        plt.ylabel('y')
        plt.legend()
        plt.figure(2)
        plt.plot(range(len(y_wer)), y_wer, label='Zbiór weryfikujący')
        plt.plot(range(len(y_wer)), predictions_wer, label='Wyjście modelu')
        plt.title(f'Predykcja dla danych weryfikujących, {"z rekurencja"if recursive else "bez rekurencji"}, liczba neuronów: {neurony}')
        plt.xlabel('k')
        plt.ylabel('y')
        plt.legend()
        print(f'Błąd średniokwadratowy dla zbioru uczącego: {mse_ucz}')
        print(f'Błąd średniokwadratowy dla zbioru weryfikującego: {mse_wer}')
        plt.show()
    return (mse_ucz, mse_wer)

if __name__ == "__main__":
    err1 = neural_network(13, 1, recursive=False)
    err1_rek = neural_network(13, 1, recursive=True)
    err6 = neural_network(13, 15, recursive=False)
    err6_rek = neural_network(13, 15, recursive=True)
    print(err1, err1_rek, err6, err6_rek)
