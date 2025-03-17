import numpy as np
from scipy.optimize import linear_sum_assignment

# Генерация матрицы C
def generate_matrix(n, mode='random', row_mode='random', col_mode='random'):
    if mode == 'random':
        C = np.random.rand(n, n) * 100  # Случайная матрица
    elif mode == 'increasing':
        C = np.array([[i * j for j in range(1, n+1)] for i in range(1, n+1)])
    elif mode == 'decreasing':
        C = np.array([[1 / (i * j) for j in range(1, n+1)] for i in range(1, n+1)])
    else:
        C = np.zeros((n, n))
    
    # Сортировка строк
    if row_mode == 'increasing':
        C = np.sort(C, axis=1)
    elif row_mode == 'decreasing':
        C = np.sort(C, axis=1)[:, ::-1]
    
    # Сортировка столбцов
    if col_mode == 'increasing':
        C = np.sort(C, axis=0)
    elif col_mode == 'decreasing':
        C = np.sort(C, axis=0)[::-1, :]
    
    return C

# Генерация вектора chi
def generate_x(n):
    return np.random.rand(n)

# Вычисление матрицы D
def calculate_D(C, chi):
    n = len(C)
    D = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            D[i, j] = sum((1 - chi[s]) * C[s, j] for s in range(j)) + \
                      (1 - chi[i]) * C[i, j] + \
                      sum(chi[s] * C[s, j] for s in range(n))
    return D

# Вычисление матрицы G_tilde
def calculate_G_tilde(C, chi):
    n = len(C)
    G_tilde = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G_tilde[i, j] = sum((1 - chi[i]) * C[i, s] for s in range(j, n))
    return G_tilde

# Жадная стратегия
def greedy_strategy(D):
    n = len(D)
    assignment = []
    used = set()
    for j in range(n):
        best_i = -1
        best_value = -np.inf
        for i in range(n):
            if i not in used and D[i, j] > best_value:
                best_value = D[i, j]
                best_i = i
        assignment.append(best_i)
        used.add(best_i)
    return assignment

# Венгерский алгоритм
def hungarian_algorithm(G_tilde):
    row_ind, col_ind = linear_sum_assignment(-G_tilde)
    return col_ind.tolist()

# Минимальная стратегия
def min_strategy(D):
    return [int(np.argmin(D[:, j])) for j in range(len(D))]

# Максимальная стратегия
def max_strategy(D):
    return [int(np.argmax(D[:, j])) for j in range(len(D))]

# Случайная стратегия
def random_strategy(D):
    n = len(D)
    return [int(np.random.choice(n)) for _ in range(n)]

# Вычисление S1
def calculate_S1(D, assignment, chi, C):
    n = len(D)
    s1 = 0
    for j in range(n):
        sigma_j = assignment[:j+1]
        # Первое слагаемое: прибыль групп с обновленной защитой
        first_term = sum((1 - chi[s]) * C[s, j] for s in sigma_j)
        # Второе слагаемое: прибыль всех групп при устаревшей защите
        second_term = sum(chi[i] * C[i, j] for i in range(n))
        s1 += first_term + second_term
    return s1

# Вычисление S2
def calculate_S2(D, assignment, chi, C):
    return sum(D[assignment[j], j] for j in range(len(D)))

# Вычисление S3
def calculate_S3(G_tilde, assignment):
    n = len(G_tilde)
    s3 = 0
    for j in range(n):
        s3 += G_tilde[assignment[j], j]
    return s3

# Основная функция для анализа
def analyze(n, mode='random', row_mode='random', col_mode='random'):
    # Генерация данных
    C = generate_matrix(n, mode, row_mode, col_mode)
    chi = generate_x(n)
    
    # Вычисление матриц
    D = calculate_D(C, chi)
    G_tilde = calculate_G_tilde(C, chi)
    
    # Применение стратегий
    greedy_assignment = greedy_strategy(D)
    hungarian_assignment = hungarian_algorithm(G_tilde)
    min_assignment = min_strategy(D)
    max_assignment = max_strategy(D)
    random_assignment = random_strategy(D)
    
    # Вычисление целевых функций
    S1_greedy = calculate_S1(D, greedy_assignment, chi, C)
    S1_min = calculate_S1(D, min_assignment, chi, C)
    S1_max = calculate_S1(D, max_assignment, chi, C)
    S1_random = calculate_S1(D, random_assignment, chi, C)
    S3_hungarian = calculate_S3(G_tilde, hungarian_assignment)
    
    # Расчёт потерь
    loss_greedy = S3_hungarian - S1_greedy
    loss_min = S3_hungarian - S1_min
    loss_max = S3_hungarian - S1_max
    loss_random = S3_hungarian - S1_random
    
    # Логирование
    print("Матрица C:")
    print(C)
    print("\nВектор chi:")
    print(chi)
    print("\nМатрица D:")
    print(D)
    print("\nМатрица G_tilde:")
    print(G_tilde)
    print("\nЖадная стратегия:", greedy_assignment, "S1:", S1_greedy, "Потери:", loss_greedy)
    print("Минимальная стратегия:", min_assignment, "S1:", S1_min, "Потери:", loss_min)
    print("Максимальная стратегия:", max_assignment, "S1:", S1_max, "Потери:", loss_max)
    print("Случайная стратегия:", random_assignment, "S1:", S1_random, "Потери:", loss_random)
    print("Венгерский алгоритм:", hungarian_assignment, "S3:", S3_hungarian)

# Пример использования
analyze(n=5, mode='random', row_mode='random', col_mode='random')