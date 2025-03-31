import numpy as np
from scipy.optimize import linear_sum_assignment
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

def generate_matrix(n, mode='random', row_mode='random', col_mode='random'):
    """Генерация матрицы C в зависимости от режима."""
    if mode == 'random':
        C = np.random.rand(n, n) * 100
    elif mode == 'increasing':
        C = np.array([[i * j for j in range(1, n+1)] for i in range(1, n+1)])
    elif mode == 'decreasing':
        C = np.array([[1/(i * j) for j in range(1, n+1)] for i in range(1, n+1)])
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

    logging.debug(f"Generated matrix C:\n{C}")
    return C

def generate_x(n):
    chi = np.random.rand(n) * 0.99  # Гарантируем, что chi_i < 1
    logging.debug(f"Generated chi vector:\n{chi}")
    return chi

def calculate_D(C, chi):
    """Вычисление матрицы D."""
    n = len(C)
    D = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            # Потери от групп с устаревшей защитой
            loss = sum(chi[s] * C[s, j] for s in range(n))
            # Прибыль от групп с обновленной защитой
            profit = sum((1 - chi[s]) * C[s, j] for s in range(j)) + (1 - chi[i]) * C[i, j]
            D[i, j] = profit + loss
    logging.debug(f"Calculated matrix D:\n{D}")
    return D

def calculate_G_tilde(C, chi):
    """Вычисление матрицы G с тильдой."""
    n = len(C)
    G_tilde = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Потери от групп с устаревшей защитой
            G_tilde[i, j] = sum(chi[i] * C[i, s] for s in range(j, n))
    logging.debug(f"Calculated matrix G_tilde:\n{G_tilde}")
    return G_tilde

def greedy_strategy(D):
    """Жадная стратегия для матрицы D."""
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
    logging.debug(f"Greedy strategy assignment: {assignment}")
    return assignment

def hungarian_algorithm(G_tilde):
    """Венгерский алгоритм для матрицы G с тильдой."""
    row_ind, col_ind = linear_sum_assignment(G_tilde)
    assignment = col_ind.tolist()
    logging.debug(f"Hungarian algorithm assignment: {assignment}")
    return assignment

def min_strategy(D):
    """Минимальная стратегия."""
    assignment = [int(np.argmin(D[:, j])) for j in range(len(D))]
    logging.debug(f"Min strategy assignment: {assignment}")
    return assignment

def max_strategy(D):
    """Максимальная стратегия."""
    assignment = [int(np.argmax(D[:, j])) for j in range(len(D))]
    logging.debug(f"Max strategy assignment: {assignment}")
    return assignment

def random_strategy(D):
    """Случайная стратегия."""
    n = len(D)
    assignment = [int(np.random.choice(n)) for _ in range(n)]
    logging.debug(f"Random strategy assignment: {assignment}")
    return assignment

def calculate_S1(D, assignment, chi, C):
    n = len(D)
    s1 = 0
    for j in range(n):
        sigma_j = assignment[:j+1]
        s1 += sum((1 - chi[s]) * C[s, j] for s in sigma_j) + sum(chi[i] * C[i, j] for i in range(n))
    return s1

def calculate_S2(D, assignment, chi, C):
    """Вычисление целевой функции S2 (Прибыль от групп с обновленной защитой)."""
    s2 = sum(D[assignment[j], j] for j in range(len(D)))
    logging.debug(f"Calculated S2: {s2}")
    return s2

def calculate_S3(G_tilde, assignment):
    n = len(G_tilde)
    s3 = 0
    for j in range(n):
        s3 += G_tilde[assignment[j], j]
    logging.debug(f"Calculated S3: {s3}")
    return s3