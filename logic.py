import numpy as np
from scipy.optimize import linear_sum_assignment

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

    return C

def generate_x(n):
    """Генерация вектора x (случайные значения от 0 до 1)."""
    return np.random.rand(n)

def calculate_D(C, chi):
    """Вычисление матрицы D."""
    n = len(C)
    D = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            D[i, j] = sum((1 - chi[s]) * C[s, j] for s in range(j)) + (1 - chi[i]) * C[i, j] + sum(chi[s] * C[s, j] for s in range(n))
    return D

def calculate_G_tilde(C, chi):
    """Вычисление матрицы G с тильдой."""
    n = len(C)
    G_tilde = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G_tilde[i, j] = sum((1 - chi[i]) * C[i, s] for s in range(j, n))
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
        assignment.append(best_i)  # Просто добавляем число, без np.int64
        used.add(best_i)
    return assignment


def hungarian_algorithm(G_tilde):
    """Венгерский алгоритм для матрицы G с тильдой."""
    row_ind, col_ind = linear_sum_assignment(-G_tilde)
    return col_ind

def min_strategy(D):
    """Минимальная стратегия."""
    return [int(np.argmin(D[:, j])) for j in range(len(D))]  # Возвращаем список чисел

def max_strategy(D):
    """Максимальная стратегия."""
    return [int(np.argmax(D[:, j])) for j in range(len(D))]  # Возвращаем список чисел

def random_strategy(D):
    """Случайная стратегия."""
    n = len(D)
    return [int(np.random.choice(n)) for _ in range(n)]  # Возвращаем список чисел

def calculate_S1(D, assignment, chi, C):
    """Вычисление целевой функции S1 (Общая прибыль с учетом всех факторов)."""
    return sum(D[assignment[j], j] for j in range(len(D))) + sum(chi[s] * C[s, j] for j in range(len(D)) for s in range(len(D)))

def calculate_S2(D, assignment, chi, C):
    """Вычисление целевой функции S2 (Прибыль от групп с обновленной защитой)."""
    return sum(D[assignment[j], j] for j in range(len(D)))

def calculate_S3(G_tilde, assignment):
    """Вычисление целевой функции S3 (Прибыль от групп с улучшенной защитой)."""
    return sum(G_tilde[assignment[j], j] for j in range(len(G_tilde)))