import numpy as np
import tkinter as tk
from tkinter import font as tkFont
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

    # Применяем дополнительные изменения для строк или столбцов
    if row_mode == 'increasing':
        C = np.sort(C, axis=1)  # Сортировка строк по возрастанию
    elif row_mode == 'decreasing':
        C = np.sort(C, axis=1)[:, ::-1]  # Сортировка строк по убыванию

    if col_mode == 'increasing':
        C = np.sort(C, axis=0)  # Сортировка столбцов по возрастанию
    elif col_mode == 'decreasing':
        C = np.sort(C, axis=0)[::-1, :]  # Сортировка столбцов по убыванию

    return C

def generate_x(n):
    """Генерация вектора x (случайные значения от 0 до 1)."""
    return np.random.rand(n)

def calculate_D(C, x):
    """Вычисление матрицы D на основе матрицы C и вектора x."""
    n = len(C)
    D = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            D[i, j] = sum((1 - x[s]) * C[s, j] for s in range(j)) + (1 - x[i]) * C[i, j] + sum(x[s] * C[s, j] for s in range(n))
    return D

def calculate_G_tilde(C, x):
    """Вычисление матрицы G с тильдой."""
    n = len(C)
    G_tilde = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G_tilde[i, j] = sum((1 - x[i]) * C[i, s] for s in range(j, n))
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

def min_strategy(D):
    """Минимальная стратегия: для каждого столбца выбираем строку с минимальным значением."""
    n = len(D)
    assignment = [np.argmin(D[:, j]) for j in range(n)]
    return assignment

def max_strategy(D):
    """Максимальная стратегия: для каждого столбца выбираем строку с максимальным значением."""
    n = len(D)
    assignment = [np.argmax(D[:, j]) for j in range(n)]
    return assignment

def random_strategy(D):
    """Случайная стратегия: для каждого столбца выбираем случайную строку."""
    n = len(D)
    assignment = [np.random.choice(range(n)) for _ in range(n)]
    return assignment

def calculate_S1(D, assignment, x, C):
    """Вычисление целевой функции S1."""
    n = len(D)
    S1_part1 = sum(D[assignment[j], j] for j in range(n))
    S1_part2 = sum(x[s] * C[s, j] for j in range(n) for s in range(n))
    return S1_part1 + S1_part2

def calculate_S2(D_tilde, assignment, x, C):
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
    row_mode = row_mode_var.get()
    col_mode = col_mode_var.get()

    # Генерация матрицы C и вектора x
    C = generate_matrix(n, mode, row_mode, col_mode)
    x = generate_x(n)

    # Вычисление матрицы D
    D = calculate_D(C, x)

    # Вычисление матрицы G с тильдой
    G_tilde = calculate_G_tilde(C, x)

    # Применение жадной стратегии к D
    greedy_assignment = greedy_strategy(D)

    # Применение Венгерского алгоритма к G_tilde
    hungarian_assignment = hungarian_algorithm(G_tilde)

    # Применение минимальной, максимальной и случайной стратегии
    min_assignment = min_strategy(D)
    max_assignment = max_strategy(D)
    random_assignment = random_strategy(D)

    # Вычисление целевых функций для всех стратегий
    S1_greedy = calculate_S1(D, greedy_assignment, x, C)
    S1_min = calculate_S1(D, min_assignment, x, C)
    S1_max = calculate_S1(D, max_assignment, x, C)
    S1_random = calculate_S1(D, random_assignment, x, C)

    S2_greedy = calculate_S2(D, greedy_assignment, x, C)
    S2_min = calculate_S2(D, min_assignment, x, C)
    S2_max = calculate_S2(D, max_assignment, x, C)
    S2_random = calculate_S2(D, random_assignment, x, C)

    S3_hungarian = calculate_S3(G_tilde, hungarian_assignment)

    # Оценка проигрыша жадной стратегии
    loss_greedy_min = S3_hungarian - S1_min
    loss_greedy_max = S3_hungarian - S1_max
    loss_greedy_random = S3_hungarian - S1_random

    # Формирование строки для вывода
    result_text = (
        f"Матрица C:\n{C}\n\n"
        f"Вектор x:\n{x}\n\n"
        f"Матрица D:\n{D}\n\n"
        f"Матрица G с тильдой:\n{G_tilde}\n\n"
        f"Жадная стратегия (назначения): {greedy_assignment}, S1 = {S1_greedy}, S2 = {S2_greedy}\n"
        f"Минимальная стратегия (назначения): {min_assignment}, S1 = {S1_min}, S2 = {S2_min}, Потери = {loss_greedy_min}\n"
        f"Максимальная стратегия (назначения): {max_assignment}, S1 = {S1_max}, S2 = {S2_max}, Потери = {loss_greedy_max}\n"
        f"Случайная стратегия (назначения): {random_assignment}, S1 = {S1_random}, S2 = {S2_random}, Потери = {loss_greedy_random}\n"
        f"Венгерский алгоритм (назначения): {hungarian_assignment}, S3 = {S3_hungarian}\n"
    )

    # Вывод результатов в текстовое поле
    text_output.delete(1.0, tk.END)  # Очищаем текстовое поле перед выводом
    text_output.insert(tk.END, result_text)  # Вставляем текст в текстовое поле

# Создание графического интерфейса
root = tk.Tk()
root.title("Анализ стратегий")
root.geometry("800x700")  # Установка размера окна

# Настройка шрифтов
font_style = tkFont.Font(family="Helvetica", size=12)

# Настройка отступов
pad_x = 10
pad_y = 5

# Элементы интерфейса
tk.Label(root, text="Размер матрицы (n):", font=font_style).grid(row=0, column=0, padx=pad_x, pady=pad_y)
entry_n = tk.Entry(root, font=font_style, width=10)
entry_n.grid(row=0, column=1, padx=pad_x, pady=pad_y)

tk.Label(root, text="Режим генерации матрицы C:", font=font_style).grid(row=1, column=0, padx=pad_x, pady=pad_y)
matrix_mode = tk.StringVar(value='random')
tk.Radiobutton(root, text="Случайная", variable=matrix_mode, value='random', font=font_style).grid(row=1, column=1, padx=pad_x, pady=pad_y)
tk.Radiobutton(root, text="Возрастающая", variable=matrix_mode, value='increasing', font=font_style).grid(row=1, column=2, padx=pad_x, pady=pad_y)
tk.Radiobutton(root, text="Убывающая", variable=matrix_mode, value='decreasing', font=font_style).grid(row=1, column=3, padx=pad_x, pady=pad_y)

tk.Label(root, text="Изменение строк:", font=font_style).grid(row=2, column=0, padx=pad_x, pady=pad_y)
row_mode_var = tk.StringVar(value='random')
tk.Radiobutton(root, text="Возрастающие", variable=row_mode_var, value='increasing', font=font_style).grid(row=2, column=1, padx=pad_x, pady=pad_y)
tk.Radiobutton(root, text="Убывающие", variable=row_mode_var, value='decreasing', font=font_style).grid(row=2, column=2, padx=pad_x, pady=pad_y)
tk.Radiobutton(root, text="Случайные", variable=row_mode_var, value='random', font=font_style).grid(row=2, column=3, padx=pad_x, pady=pad_y)

tk.Label(root, text="Изменение столбцов:", font=font_style).grid(row=3, column=0, padx=pad_x, pady=pad_y)
col_mode_var = tk.StringVar(value='random')
tk.Radiobutton(root, text="Возрастающие", variable=col_mode_var, value='increasing', font=font_style).grid(row=3, column=1, padx=pad_x, pady=pad_y)
tk.Radiobutton(root, text="Убывающие", variable=col_mode_var, value='decreasing', font=font_style).grid(row=3, column=2, padx=pad_x, pady=pad_y)
tk.Radiobutton(root, text="Случайные", variable=col_mode_var, value='random', font=font_style).grid(row=3, column=3, padx=pad_x, pady=pad_y)

tk.Button(root, text="Запустить анализ", command=run_analysis, font=font_style, width=20).grid(row=4, column=0, columnspan=4, pady=pad_y)

# Текстовое поле для вывода результатов
text_output = tk.Text(root, height=40, width=80, font=font_style)  # Увеличена высота окна вывода
text_output.grid(row=5, column=0, columnspan=4, padx=pad_x, pady=pad_y)

root.mainloop()
