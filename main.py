import numpy as np
import tkinter as tk
from tkinter import messagebox
from scipy.optimize import linear_sum_assignment

def generate_matrix(n, mode='random'):
    """Генерация матрицы C в зависимости от режима."""
    if mode == 'random':
        return np.random.rand(n, n) * 100
    elif mode == 'increasing':
        return np.array([[i * j for j in range(1, n+1)] for i in range(1, n+1)])
    elif mode == 'decreasing':
        return np.array([[1/(i * j) for j in range(1, n+1)] for i in range(1, n+1)])
    else:
        return np.zeros((n, n))

def generate_chi(n):
    """Генерация вектора chi (случайные значения от 0 до 1)."""
    return np.random.rand(n)

def calculate_D(C, chi):
    """Вычисление матрицы D на основе матрицы C и вектора chi."""
    n = len(C)
    D = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            D[i, j] = sum((1 - chi[s]) * C[s, j] for s in range(j)) + (1 - chi[i]) * C[i, j] + sum(chi[s] * C[s, j] for s in range(n))
    return D

def calculate_G_tilde(C, chi):
    """Вычисление матрицы G с тильдой (формула 24 из пособия)."""
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
        assignment.append(best_i)
        used.add(best_i)
    return assignment

def hungarian_algorithm(G_tilde):
    """Венгерский алгоритм для матрицы G с тильдой."""
    row_ind, col_ind = linear_sum_assignment(-G_tilde)
    return col_ind

def calculate_S1(D, assignment, chi, C):
    """Вычисление целевой функции S1."""
    n = len(D)
    # Первая часть S1: сумма d_{sigma(j), j} для назначенных групп
    S1_part1 = sum(D[assignment[j], j] for j in range(n))
    # Вторая часть S1: сумма chi_s * c_sj для всех s и j
    S1_part2 = sum(chi[s] * C[s, j] for j in range(n) for s in range(n))
    return S1_part1 + S1_part2

def calculate_S2(D_tilde, assignment, chi, C):
    """Вычисление целевой функции S2 для матрицы D_tilde."""
    n = len(D_tilde)
    S2 = sum(D_tilde[assignment[j], j] for j in range(n))
    return S2

def calculate_S3(G_tilde, assignment):
    """Вычисление целевой функции S3 для матрицы G_tilde."""
    n = len(G_tilde)
    S3 = sum(G_tilde[assignment[j], j] for j in range(n))
    return S3

def run_analysis():
    """Запуск анализа."""
    n = int(entry_n.get())
    mode = matrix_mode.get()
    
    # Генерация матрицы C и вектора chi
    C = generate_matrix(n, mode)
    chi = generate_chi(n)
    
    # Вычисление матрицы D
    D = calculate_D(C, chi)
    
    # Вычисление матрицы G с тильдой
    G_tilde = calculate_G_tilde(C, chi)
    
    # Применение жадной стратегии к D
    greedy_assignment = greedy_strategy(D)
    
    # Применение Венгерского алгоритма к G_tilde
    hungarian_assignment = hungarian_algorithm(G_tilde)
    
    # Вычисление целевых функций для обеих стратегий
    S1_greedy = calculate_S1(D, greedy_assignment, chi, C)
    S2_greedy = calculate_S2(D, greedy_assignment, chi, C)
    S3_hungarian = calculate_S3(G_tilde, hungarian_assignment)
    
    # Оценка проигрыша жадной стратегии
    loss = S3_hungarian - S1_greedy
    
    # Вывод результатов
    result_text = (
        f"Матрица C:\n{C}\n\n"
        f"Вектор chi:\n{chi}\n\n"
        f"Матрица D:\n{D}\n\n"
        f"Матрица G с тильдой:\n{G_tilde}\n\n"
        f"Жадная стратегия (назначения): {greedy_assignment}, S1 = {S1_greedy}, S2 = {S2_greedy}\n"
        f"Венгерский алгоритм (назначения): {hungarian_assignment}, S3 = {S3_hungarian}\n\n"
        f"Оценка проигрыша жадной стратегии: {loss}\n"
    )
    
    messagebox.showinfo("Результаты анализа", result_text)

# Создание графического интерфейса
root = tk.Tk()
root.title("Анализ стратегий")

tk.Label(root, text="Размер матрицы (n):").grid(row=0, column=0)
entry_n = tk.Entry(root)
entry_n.grid(row=0, column=1)

tk.Label(root, text="Режим генерации матрицы C:").grid(row=1, column=0)
matrix_mode = tk.StringVar(value='random')
tk.Radiobutton(root, text="Случайная", variable=matrix_mode, value='random').grid(row=1, column=1)
tk.Radiobutton(root, text="Возрастающая", variable=matrix_mode, value='increasing').grid(row=1, column=2)
tk.Radiobutton(root, text="Убывающая", variable=matrix_mode, value='decreasing').grid(row=1, column=3)

tk.Button(root, text="Запустить анализ", command=run_analysis).grid(row=2, column=0, columnspan=4)

root.mainloop()
