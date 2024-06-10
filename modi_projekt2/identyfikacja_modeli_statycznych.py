import numpy as np
import matplotlib.pyplot as plt

###############    ZADANIE 1 PUNKT a       ##################

# pobranie danych i podzielenie ich na zbiory
with open('danestat33.txt', 'r') as file:
    u = []
    y = []
    for line in file:
        columns = line.split()
        u.append(float(columns[0]))
        y.append(float(columns[1]))

u_ucz = [u[i] for i in range(len(u)) if i % 2 == 0]
u_wer = [u[i] for i in range(len(u)) if i % 2 != 0]
y_ucz = [y[i] for i in range(len(y)) if i % 2 == 0]
y_wer = [y[i] for i in range(len(y)) if i % 2 != 0]
u_ucz = np.array(u_ucz)
y_ucz = np.array(y_ucz)
u_wer = np.array(u_wer)
y_wer = np.array(y_wer)
u = np.array(u)
y = np.array(y)

# Plot zbiorów
def plot_zad1_a():
    plt.scatter(u, y)
    plt.xlabel('u')
    plt.ylabel('y')
    plt.title('All Static Data')
    plt.show()

    plt.scatter(u_ucz, y_ucz)
    plt.xlabel('u')
    plt.ylabel('y')
    plt.title('Static Training Data')
    plt.show()

    plt.scatter(u_wer, y_wer, color='orange')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.title('Static Validation Data')
    plt.show()

###############    ZADANIE 1 PUNKT B       ##################
def zad1_b(u_ucz, u_wer, y_ucz, y_wer):
    u_ucz = u_ucz.reshape(-1, 1)
    y_ucz = y_ucz.reshape(-1, 1)
    u_wer = u_wer.reshape(-1, 1)
    y_wer = y_wer.reshape(-1, 1)

    # stworzenie M_ucz i M_wer
    M_ucz = np.column_stack((np.ones_like(u_ucz), u_ucz))
    M_wer = np.column_stack((np.ones_like(u_wer), u_wer))
    M_stat = np.column_stack((np.ones_like(u), u))

    # Do obliczenia współczynników użyto odpowiednika lewego dzielenia z MATLABA
    wsp_ucz = np.linalg.lstsq(M_ucz, y_ucz, rcond=None)[0]

    # Macierze Y obliczone przy użyciu współczynników 
    y_ucz_hat = np.dot(M_ucz, wsp_ucz)
    y_wer_hat = np.dot(M_wer, wsp_ucz)
    y_stat = np.dot(M_stat, wsp_ucz)

    # plot statycznej
    plt.plot(u, y_stat, label=f'y(u)')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.title('Characteristic y(u)')
    plt.legend()
    plt.show()

    # Plot dla trenujących
    plt.scatter(u_ucz, y_ucz, label='Training Data')
    plt.plot(u_ucz, y_ucz_hat, color='red', label=f'Model Output: y(u) = {wsp_ucz[0][0]:.2f} + {wsp_ucz[1][0]:.2f}u')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.title('Model Output vs Training Data')
    plt.legend()
    plt.show()

    # Plot dla weryfikujących
    plt.scatter(u_wer, y_wer, color='orange', label='Verification Data')
    plt.plot(u_wer, y_wer_hat, color='red', label=f'Model Output: y(u) = {wsp_ucz[0][0]:.2f} + {wsp_ucz[1][0]:.2f}u')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.title('Model Output vs Verification Data')
    plt.legend()
    plt.show()

    # Obliczanie błędów kwadratowych
    error_ucz = np.sum((y_ucz - y_ucz_hat) ** 2)
    error_wer = np.sum((y_wer - y_wer_hat) ** 2)
    print(error_ucz, error_wer)


###############    ZADANIE 1 PUNKT C       ##################
def zad1_c(u, y, u_ucz, u_wer, y_ucz, y_wer):
    u_ucz = u_ucz.reshape(-1, 1)
    y_ucz = y_ucz.reshape(-1, 1)
    u_wer = u_wer.reshape(-1, 1)
    y_wer = y_wer.reshape(-1, 1)
    u = u.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Stopnie wielomianu
    N_values = [1, 2, 3, 4, 5, 6]

    errors_ucz = []
    errors_wer = []

    for N in N_values:
        M_ucz = np.column_stack([u_ucz ** i for i in range(N + 1)])
        M_wer = np.column_stack([u_wer ** i for i in range(N + 1)])
        M_stat = np.column_stack([u ** i for i in range(N + 1)])

        wsp_ucz = np.linalg.lstsq(M_ucz, y_ucz, rcond=None)[0]

        y_ucz_hat = np.dot(M_ucz, wsp_ucz)
        y_wer_hat = np.dot(M_wer, wsp_ucz)
        y_stat = np.dot(M_stat, wsp_ucz)

        # plot statycznej
        plt.scatter(u, y_stat, label=f'y(u)')
        plt.xlabel('u')
        plt.ylabel('y')
        plt.title(f'Characteristic y(u), Degree {N}')
        plt.legend()
        plt.show()

        # Plot uczących
        plt.scatter(u_ucz, y_ucz, label='Training Data', alpha=0.5)
        plt.scatter(u_ucz, y_ucz_hat, color='red', label=f'Model Output: Degree {N}', alpha=0.8, s=20)
        plt.xlabel('u')
        plt.ylabel('y')
        plt.title(f'Model Output vs Training Data: Degree {N}')
        plt.legend()
        plt.show()

        # Plot weryfikujących
        plt.scatter(u_wer, y_wer, label='Verification Data', alpha=0.5)
        plt.scatter(u_wer, y_wer_hat, color='green', label=f'Model Output: Degree {N}', alpha=0.8, s=20)
        plt.xlabel('u')
        plt.ylabel('y')
        plt.title(f'Model Output vs Verification Data: Degree {N}')
        plt.legend()
        plt.show()
    
        # Błędy kwadratowe
        error_ucz = np.sum((y_ucz - y_ucz_hat) ** 2)
        error_wer = np.sum((y_wer - y_wer_hat) ** 2)
        
        errors_ucz.append(error_ucz)
        errors_wer.append(error_wer)

    print("Polynomial Degree | Training Error | Verification Error")
    for N, error_ucz, error_wer in zip(N_values, errors_ucz, errors_wer):
        print(f"{N:17} | {error_ucz:.6f}      | {error_wer:.6f}")



if __name__ == "__main__":
    plot_zad1_a()
    zad1_b(u_ucz, u_wer, y_ucz, y_wer)
    zad1_c(u, y, u_ucz, u_wer, y_ucz, y_wer)