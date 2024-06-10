import numpy as np
import matplotlib.pyplot as plt

###############    ZADANIE 2 PUNKT a       ##################

# pobranie danych i podzielenie ich na zbiory
with open('danedynucz33.txt', 'r') as file:
    u_ucz = []
    y_ucz = []
    for line in file:
        columns = line.split()
        u_ucz.append(float(columns[0]))
        y_ucz.append(float(columns[1]))

with open('danedynwer33.txt', 'r') as file:
    u_wer = []
    y_wer = []
    for line in file:
        columns = line.split()
        u_wer.append(float(columns[0]))
        y_wer.append(float(columns[1]))

u_ucz = np.array(u_ucz)
y_ucz = np.array(y_ucz)
u_wer = np.array(u_wer)
y_wer = np.array(y_wer)

def zad2_a():
    plt.plot(u_ucz, label='u_train')
    plt.plot(y_ucz, label='y_train', color='orange')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.legend()
    plt.title('Dynamic training Data')
    plt.show()

    plt.plot(u_wer, label='u_verification')
    plt.plot(y_wer, label='y_verification', color='orange')
    plt.xlabel('k')
    plt.ylabel('y')
    plt.legend()
    plt.title('Dynamic validation Data')
    plt.show()

###############    ZADANIE 2 PUNKT b       ##################

def zad2_b(u_ucz, u_wer, y_ucz, y_wer, n):
    u_ucz = u_ucz.reshape(-1, 1)
    y_ucz = y_ucz.reshape(-1, 1)
    u_wer = u_wer.reshape(-1, 1)
    y_wer = y_wer.reshape(-1, 1)


    if n == 1:
        y_ucz_hat = []
        y_wer_hat = []
        y_ucz_hat_rek = []
        y_wer_hat_rek = []

        y_ucz_hat.append(y_ucz[0])
        y_ucz_hat_rek.append(y_ucz[0])        
        y_wer_hat.append(y_ucz[0])
        y_wer_hat_rek.append(y_ucz[0])   

        M_ucz = np.column_stack((u_ucz[:-1], y_ucz[:-1]))
        wsp_ucz = np.linalg.lstsq(M_ucz, y_ucz[1:], rcond=None)[0]

        for i in range(len(y_ucz)-n):
            y_ucz_hat.append(u_ucz[i]*wsp_ucz[0] + y_ucz[i]*wsp_ucz[1])
            y_wer_hat.append(u_wer[i]*wsp_ucz[0] + y_wer[i]*wsp_ucz[1])
            y_ucz_hat_rek.append(u_ucz[i]*wsp_ucz[0] + y_ucz_hat_rek[i]*wsp_ucz[1])
            y_wer_hat_rek.append(u_wer[i]*wsp_ucz[0] + y_wer_hat_rek[i]*wsp_ucz[1])

    if n == 2:
        y_ucz_hat = []
        y_wer_hat = []
        y_ucz_hat_rek = []
        y_wer_hat_rek = []

        y_ucz_hat.append(y_ucz[0])
        y_ucz_hat_rek.append(y_ucz[0])        
        y_wer_hat.append(y_ucz[0])
        y_wer_hat_rek.append(y_ucz[0])
        y_ucz_hat.append(y_ucz[1])
        y_ucz_hat_rek.append(y_ucz[1])        
        y_wer_hat.append(y_ucz[1])
        y_wer_hat_rek.append(y_ucz[1])

        M_ucz = np.column_stack((u_ucz[1:-1], y_ucz[1:-1], u_ucz[:-2], y_ucz[:-2]))
        wsp_ucz = np.linalg.lstsq(M_ucz, y_ucz[2:], rcond=None)[0]

        for i in range(len(y_ucz)-n):
            y_ucz_hat.append(u_ucz[i+1]*wsp_ucz[0] + y_ucz[i+1]*wsp_ucz[1] + u_ucz[i]*wsp_ucz[2] + y_ucz[i]*wsp_ucz[3])
            y_wer_hat.append(u_wer[i+1]*wsp_ucz[0] + y_wer[i+1]*wsp_ucz[1] + u_wer[i]*wsp_ucz[2] + y_wer[i]*wsp_ucz[3])
            y_ucz_hat_rek.append(u_ucz[i+1]*wsp_ucz[0] + y_ucz_hat_rek[i+1]*wsp_ucz[1] + u_ucz[i]*wsp_ucz[2] + y_ucz_hat_rek[i]*wsp_ucz[3])
            y_wer_hat_rek.append(u_wer[i+1]*wsp_ucz[0] + y_wer_hat_rek[i+1]*wsp_ucz[1] + u_wer[i]*wsp_ucz[2] + y_wer_hat_rek[i]*wsp_ucz[3])

    if n == 3:
        y_ucz_hat = []
        y_wer_hat = []
        y_ucz_hat_rek = []
        y_wer_hat_rek = []

        y_ucz_hat.append(y_ucz[0])
        y_ucz_hat_rek.append(y_ucz[0])        
        y_wer_hat.append(y_ucz[0])
        y_wer_hat_rek.append(y_ucz[0])
        y_ucz_hat.append(y_ucz[1])
        y_ucz_hat_rek.append(y_ucz[1])        
        y_wer_hat.append(y_ucz[1])
        y_wer_hat_rek.append(y_ucz[2])
        y_ucz_hat.append(y_ucz[2])
        y_ucz_hat_rek.append(y_ucz[2])        
        y_wer_hat.append(y_ucz[2])
        y_wer_hat_rek.append(y_ucz[2])
    
        M_ucz = np.column_stack((u_ucz[2:-1], y_ucz[2:-1], u_ucz[1:-2], y_ucz[1:-2], u_ucz[:-3], y_ucz[:-3]))
        wsp_ucz = np.linalg.lstsq(M_ucz, y_ucz[3:], rcond=None)[0]
       
        for i in range(len(y_ucz)-n):
            y_ucz_hat.append(u_ucz[i+2]*wsp_ucz[0] + y_ucz[i+2]*wsp_ucz[1] + u_ucz[i+1]*wsp_ucz[2] + y_ucz[i+1]*wsp_ucz[3] + u_ucz[i]*wsp_ucz[4] + y_ucz[i]*wsp_ucz[5])
            y_wer_hat.append(u_wer[i+2]*wsp_ucz[0] + y_wer[i+2]*wsp_ucz[1] + u_wer[i+1]*wsp_ucz[2] + y_wer[i+1]*wsp_ucz[3] + u_wer[i]*wsp_ucz[4] + y_wer[i]*wsp_ucz[5])
            y_ucz_hat_rek.append(u_ucz[i+2]*wsp_ucz[0] + y_ucz_hat_rek[i+2]*wsp_ucz[1]  + u_ucz[i+1]*wsp_ucz[2] + y_ucz_hat_rek[i+1]*wsp_ucz[3] + u_ucz[i]*wsp_ucz[4] + y_ucz_hat_rek[i]*wsp_ucz[5])
            y_wer_hat_rek.append(u_wer[i+2]*wsp_ucz[0] + y_wer_hat_rek[i+2]*wsp_ucz[1] + u_wer[i+1]*wsp_ucz[2] + y_wer_hat_rek[i+1]*wsp_ucz[3] + u_wer[i]*wsp_ucz[4] + y_wer_hat_rek[i]*wsp_ucz[5])


    # obliczanie i wypisywanie błedu
    error_ucz = np.sum((y_ucz - y_ucz_hat) ** 2)
    error_wer = np.sum((y_wer - y_wer_hat) ** 2)
    error_ucz_rek = np.sum((y_ucz - y_ucz_hat_rek) ** 2)
    error_wer_rek = np.sum((y_wer - y_wer_hat_rek) ** 2)
    print(error_ucz, error_wer)
    print(error_ucz_rek, error_wer_rek)

    # plot uczacych
    plt.figure(figsize=(10, 6))
    plt.plot(y_ucz, label='Actual Output (Training)', color='blue')
    plt.plot(y_ucz_hat, label='Model Output (Training)', linestyle='--', color='red')
    plt.xlabel('k')
    plt.ylabel('Output (y)')
    plt.title(f'Training Data no recursion, r. {n}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot weryfikujacych
    plt.figure(figsize=(10, 6))
    plt.plot(y_wer, label='Actual Output (Verification)', color='blue')
    plt.plot(y_wer_hat, label='Model Output (Verification)', linestyle='--', color='red')
    plt.xlabel('k')
    plt.ylabel('Output (y)')
    plt.title(f'Verifying Data no recursion, r. {n}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # plot uczacych dla rekurencyjnego
    plt.figure(figsize=(10, 6))
    plt.plot(y_ucz, label='Actual Output (Training)', color='blue')
    plt.plot(y_ucz_hat_rek, label='Model Output (Training)', linestyle='--', color='red')
    plt.xlabel('k')
    plt.ylabel('Output (y)')
    plt.title(f'Training Data with recursion, r. {n}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot weryfikujacych dla rekurencyjnego
    plt.figure(figsize=(10, 6))
    plt.plot(y_wer, label='Actual Output (Verification)', color='blue')
    plt.plot(y_wer_hat_rek, label='Model Output (Verification)', linestyle='--', color='red')
    plt.xlabel('k')
    plt.ylabel('Output (y)')
    plt.title(f'Verifying Data with recursion, r. {n}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


###############    ZADANIE 2 PUNKT c       ##################
def zad2_c(u_ucz, u_wer, y_ucz, y_wer, n_dyn, n_poly):
    y_ucz_hat = []
    y_wer_hat = [] 
    y_ucz_hat_rek = []
    y_wer_hat_rek = []

    for i in range(n_dyn):
        y_ucz_hat.append(y_ucz[i])
        y_wer_hat.append(y_ucz[i])
        y_ucz_hat_rek.append(y_ucz[i])
        y_wer_hat_rek.append(y_ucz[i])

    columns = []
    for i in range(1, n_dyn+1):
        for j in range(1, n_poly+1):
            columns.append(u_ucz[n_dyn-i:-i]**j)
            columns.append(y_ucz[n_dyn-i:-i]**j)
    M_ucz = np.column_stack(tuple(columns))
    wsp_ucz = np.linalg.lstsq(M_ucz, y_ucz[n_dyn:], rcond=None)[0]
    for i in range(len(y_ucz)-n_dyn):
        yk = 0
        yk_wer = 0
        yk_rek = 0
        yk_wer_rek = 0
        counter = 0
        for j in range(1, n_dyn+1):
            for k in range(1, n_poly+1):
                yk += wsp_ucz[counter]*u_ucz[i+n_dyn-j]**k + wsp_ucz[counter+1]*y_ucz[i+n_dyn-j]**k
                yk_wer += wsp_ucz[counter]*u_wer[i+n_dyn-j]**k + wsp_ucz[counter+1]*y_wer[i+n_dyn-j]**k
                yk_rek += wsp_ucz[counter]*u_ucz[i+n_dyn-j]**k + wsp_ucz[counter+1]*y_ucz_hat_rek[i+n_dyn-j]**k
                yk_wer_rek += wsp_ucz[counter]*u_wer[i+n_dyn-j]**k + wsp_ucz[counter+1]*y_wer_hat_rek[i+n_dyn-j]**k
                counter += 2
        y_ucz_hat.append(yk)
        y_wer_hat.append(yk_wer)
        y_ucz_hat_rek.append(yk_rek)
        y_wer_hat_rek.append(yk_wer_rek)

    # obliczanie i wypisywanie błedu
    
    error_ucz = np.sum((y_ucz - y_ucz_hat) ** 2)
    error_wer = np.sum((y_wer - y_wer_hat) ** 2)
    error_ucz_rek = np.sum((y_ucz - y_ucz_hat_rek) ** 2)
    error_wer_rek = np.sum((y_wer - y_wer_hat_rek) ** 2)
    #return error_ucz, error_wer, error_ucz_rek, error_wer_rek
    print(error_ucz, error_wer)
    print(error_ucz_rek, error_wer_rek)


    # plot uczacych
    plt.figure(figsize=(10, 6))
    plt.plot(y_ucz[:-n_dyn], label='Actual Output (Training)', color='blue')
    plt.plot(y_ucz_hat, label='Model Output (Training)', linestyle='--', color='red')
    plt.xlabel('k')
    plt.ylabel('Output (y)')
    plt.title(f'Training Data no recursion, r. {n_dyn}, deg. poly. {n_poly}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot weryfikujacych
    plt.figure(figsize=(10, 6))
    plt.plot(y_wer[:-n_dyn], label='Actual Output (Verification)', color='blue')
    plt.plot(y_wer_hat, label='Model Output (Verification)', linestyle='--', color='red')
    plt.xlabel('k')
    plt.ylabel('Output (y)')
    plt.title(f'Verifying Data no recursion, r. {n_dyn}, deg. poly. {n_poly}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # plot uczacych dla rekurencyjnego
    plt.figure(figsize=(10, 6))
    plt.plot(y_ucz, label='Actual Output (Training)', color='blue')
    plt.plot(y_ucz_hat_rek, label='Model Output (Training)', linestyle='--', color='red')
    plt.xlabel('k')
    plt.ylabel('Output (y)')
    plt.title(f'Training Data with recursion, r. {n_dyn}, deg. poly. {n_poly}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot weryfikujacych dla rekurencyjnego
    plt.figure(figsize=(10, 6))
    plt.plot(y_wer, label='Actual Output (Verification)', color='blue')
    plt.plot(y_wer_hat_rek, label='Model Output (Verification)', linestyle='--', color='red')
    plt.xlabel('k')
    plt.ylabel('Output (y)')
    plt.title(f'Verifying Data with recursion, r. {n_dyn}, deg. poly. {n_poly}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

###############    ZADANIE 2 PUNKT d       ##################
def zad2_d(u_ucz, u_wer, y_ucz, y_wer, n_dyn=13, n_poly=5):
    y_wer_hat_rek = []
    u_wer = np.linspace(-1, 1, 2000)

    for i in range(n_dyn):
        y_wer_hat_rek.append(y_ucz[i])

    columns = []
    for i in range(1, n_dyn+1):
        for j in range(1, n_poly+1):
            columns.append(u_ucz[n_dyn-i:-i]**j)
            columns.append(y_ucz[n_dyn-i:-i]**j)
    M_ucz = np.column_stack(tuple(columns))
    wsp_ucz = np.linalg.lstsq(M_ucz, y_ucz[n_dyn:], rcond=None)[0]
    for i in range(len(y_ucz)-n_dyn):
        yk_wer_rek = 0
        counter = 0
        for j in range(1, n_dyn+1):
            for k in range(1, n_poly+1):
                yk_wer_rek += wsp_ucz[counter]*u_wer[i+n_dyn-j]**k + wsp_ucz[counter+1]*y_wer_hat_rek[i+n_dyn-j]**k
                counter += 2
        y_wer_hat_rek.append(yk_wer_rek)

    # plot statycznej dla rekurencyjnego
    plt.figure(figsize=(10, 6))
    plt.plot(u_wer, y_wer_hat_rek, label='y(u)', linestyle='--', color='red')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.title('Model statyczny otrzymany metodą symulacyjną')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # zad2_a()

    # for i in range(1, 4):
    #     print(i)
    #     zad2_b(u_ucz, u_wer, y_ucz, y_wer, i)
    #     print()    

    #zad2_c(u_ucz, u_wer, y_ucz, y_wer, 13, 5)

    # for i in range(1,4):
    #     for j in range(1,4):
    #         print(i,j)
    #         zad2_c(u_ucz, u_wer, y_ucz, y_wer, i, j)
    #         print()
    
    zad2_d(u_ucz, u_wer, y_ucz, y_wer)

    # best_error_rek = float("inf")
    # best_rek = []
    # best_error = float("inf")
    # best = []
    # for i in range(1,30):
    #     for j in range(1,6):
    #         error_ucz, error_wer, error_ucz_rek, error_wer_rek = zad2_c(u_ucz, u_wer, y_ucz, y_wer, i, j)
    #         if error_wer < best_error:
    #             best_error = error_wer
    #             best = (i,j)
    #         if error_wer_rek < best_error_rek:
    #             best_error_rek = error_wer_rek
    #             best_rek = (i,j)
    # print(best)
    # print(best_rek)