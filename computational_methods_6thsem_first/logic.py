import numpy as np
from scipy.optimize import linear_sum_assignment

# Генерация матрицы C
def generate_matrix(n, mode='random', row_mode='random', col_mode='random'):
    if mode == 'random':
        C = np.random.rand(n, n) * 100
    elif mode == 'increasing':
        C = np.array([[i * j for j in range(1, n+1)] for i in range(1, n+1)])
    elif mode == 'decreasing':
        C = np.array([[1 / (i * j) for j in range(1, n+1)] for i in range(1, n+1)])
    else:
        C = np.zeros((n, n))
    
    if row_mode == 'increasing':
        C = np.sort(C, axis=1)
    elif row_mode == 'decreasing':
        C = np.sort(C, axis=1)[:, ::-1]
    
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

def calculate_D_tilde(C, assignment, chi):
    n = len(C)
    D_tilde = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Назначения до текущего j (не включая)
            assigned_before_j = [assignment[k] for k in range(j) if k < len(assignment)]
            first_term = sum((1 - chi[s]) * C[s, j] for s in assigned_before_j)
            second_term = (1 - chi[i]) * C[i, j]
            D_tilde[i, j] = first_term + second_term
    return D_tilde

# Вычисление матрицы G_tilde
def calculate_G_tilde(C, chi):
    n = len(C)
    G_tilde = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            G_tilde[i,j] = sum((1-chi[i])*C[i,s] for s in range(j,n))
    return G_tilde

# Жадная стратегия
def greedy_strategy(G):
    """Жадная стратегия (возвращает сумму и назначения)"""
    n = len(G)
    total = 0
    used = set()
    assignments = []
    
    for j in range(n):
        valid = [(i, G[i,j]) for i in range(n) if i not in used]
        if not valid:  # если все уже использованы (не должно происходить)
            valid = [(i, G[i,j]) for i in range(n) if i not in used]
        best_i, best_val = max(valid, key=lambda x: x[1])
        total += best_val
        used.add(best_i)
        assignments.append(best_i)
    
    return assignments

# Венгерский алгоритм
def hungarian_algorithm(G):
    """Венгерский алгоритм для максимизации (возвращает сумму и назначения)"""
    row_ind, col_ind = linear_sum_assignment(-G)
    total = sum(G[row_ind[i], col_ind[i]] for i in range(len(row_ind)))
    # Сортируем назначения по порядку столбцов
    assignments = [0] * len(row_ind)
    for i, j in zip(row_ind, col_ind):
        assignments[j] = i
    return assignments

# Минимальная стратегия
def min_strategy(G):
    """Бережливая стратегия (возвращает сумму и назначения)"""
    n = len(G)
    total = 0
    used = set()
    assignments = []
    
    for j in range(n):
        valid = [(i, G[i,j]) for i in range(n) if i not in used]
        if not valid:  # если все уже использованы (не должно происходить)
            valid = [(i, G[i,j]) for i in range(n) if i not in used]
        best_i, best_val = min(valid, key=lambda x: x[1])
        total += best_val
        used.add(best_i)
        assignments.append(best_i)
    
    return assignments

# Случайная стратегия
def random_strategy(G):
    n = len(G)
    assignment = []
    available = list(range(n))
    for j in range(n):
        if not available:
            available = [i for i in range(n) if i not in assignment]
        chosen = np.random.choice(available)
        assignment.append(chosen)
        available.remove(chosen)
    return assignment

# Проверка назначений
def validate_assignment(assignment, n):
    if len(set(assignment)) != n:
        print(f"Invalid assignment: {assignment}")
        return False
    return True

# Вычисление S1
def calculate_S1(D, assignment, chi, C):
    n = len(D)
    common_term = sum(chi[i] * C[i, j] for j in range(n) for i in range(n))
    s1 = 0
    for j in range(n):
        sigma_j = assignment[:j]
        first_term = sum((1 - chi[s]) * C[s, j] for s in sigma_j)
        s1 += first_term
    return s1 + common_term

# Вычисление S2
def calculate_S2(D_tilde, assignment):
    return sum(D_tilde[assignment[j], j] for j in range(len(D_tilde)))

# Вычисление S3
def calculate_S3(G_tilde, assignment):
    return sum(G_tilde[assignment[j], j] for j in range(len(G_tilde)))

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
    random_assignment = random_strategy(D)
    
    # Вычисление целевых функций
    S2_greedy = calculate_S2(calculate_D_tilde(C, greedy_assignment, chi), greedy_assignment)
    S2_min = calculate_S2(calculate_D_tilde(C, min_assignment, chi), min_assignment)
    S2_random = calculate_S2(calculate_D_tilde(C, random_assignment, chi), random_assignment)
    S3_hungarian = calculate_S3(G_tilde, hungarian_assignment)
    
    # Расчёт потерь
    loss_greedy = S3_hungarian - S2_greedy
    loss_min = S3_hungarian - S2_min
    loss_random = S3_hungarian - S2_random
    
    # Логирование
    print("Матрица C:")
    print(C)
    print("\nВектор chi:")
    print(chi)
    print("\nМатрица D:")
    print(D)
    print("\nМатрица G_tilde:")
    print(G_tilde)
    print("\nЖадная стратегия:", greedy_assignment, "S2:", S2_greedy, "Потери:", loss_greedy)
    print("Минимальная стратегия:", min_assignment, "S2:", S2_min, "Потери:", loss_min)
    print("Случайная стратегия:", random_assignment, "S2:", S2_random, "Потери:", loss_random)
    print("Венгерский алгоритм:", hungarian_assignment, "S3:", S3_hungarian)

# Пример использования
analyze(n=50, mode='random', row_mode='random', col_mode='random')