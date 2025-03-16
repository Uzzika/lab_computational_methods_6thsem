import numpy as np
import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageTk  # Для работы с изображениями

# Функции для работы с матрицами и стратегиями
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

def calculate_D(C, x, chi):
    """Вычисление матрицы D на основе матрицы C, вектора x и коэффициента chi."""
    n = len(C)
    D = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            D[i, j] = sum((1 - chi[s]) * C[s, j] for s in range(j)) + (1 - chi[i]) * C[i, j] + sum(chi[s] * C[s, j] for s in range(n))
    return D

def calculate_G_tilde(C, x, chi):
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

def calculate_S1(D, assignment, x, C, chi):
    """Вычисление целевой функции S1 с учетом коэффициента chi."""
    n = len(D)
    S1_part1 = sum(D[assignment[j], j] for j in range(n))
    S1_part2 = sum(x[s] * C[s, j] for j in range(n) for s in range(n))
    return S1_part1 + S1_part2

def calculate_S2(D_tilde, assignment, x, C, chi):
    """Вычисление целевой функции S2 для матрицы D_tilde."""
    n = len(D_tilde)
    S2 = sum(D_tilde[assignment[j], j] for j in range(n))
    return S2

def calculate_S3(G_tilde, assignment):
    """Вычисление целевой функции S3 для матрицы G_tilde."""
    n = len(G_tilde)
    S3 = sum(G_tilde[assignment[j], j] for j in range(n))
    return S3

# Запуск анализа
def run_analysis():
    """Запуск анализа."""
    n = int(entry_n.get())
    mode = matrix_mode.get()
    row_mode = row_mode_var.get()
    col_mode = col_mode_var.get()

    # Генерация матрицы C и вектора x
    C = generate_matrix(n, mode, row_mode, col_mode)
    x = generate_x(n)
    chi = np.random.rand(n)  # Генерация коэффициентов chi

    # Вычисление матрицы D
    D = calculate_D(C, x, chi)

    # Вычисление матрицы G с тильдой
    G_tilde = calculate_G_tilde(C, x, chi)

    # Применение жадной стратегии к D
    greedy_assignment = greedy_strategy(D)

    # Применение Венгерского алгоритма к G_tilde
    hungarian_assignment = hungarian_algorithm(G_tilde)

    # Применение минимальной, максимальной и случайной стратегии
    min_assignment = min_strategy(D)
    max_assignment = max_strategy(D)
    random_assignment = random_strategy(D)

    # Вычисление целевых функций для всех стратегий
    S1_greedy = calculate_S1(D, greedy_assignment, x, C, chi)
    S1_min = calculate_S1(D, min_assignment, x, C, chi)
    S1_max = calculate_S1(D, max_assignment, x, C, chi)
    S1_random = calculate_S1(D, random_assignment, x, C, chi)

    S2_greedy = calculate_S2(D, greedy_assignment, x, C, chi)
    S2_min = calculate_S2(D, min_assignment, x, C, chi)
    S2_max = calculate_S2(D, max_assignment, x, C, chi)
    S2_random = calculate_S2(D, random_assignment, x, C, chi)

    S3_hungarian = calculate_S3(G_tilde, hungarian_assignment)

    # Оценка проигрыша жадной стратегии
    loss_greedy_min = S3_hungarian - S1_min
    loss_greedy_max = S3_hungarian - S1_max
    loss_greedy_random = S3_hungarian - S1_random

    # Формирование строки для вывода
    result_text = (
        f"Жадная стратегия (назначения): {greedy_assignment}, S1 = {S1_greedy}, S2 = {S2_greedy}\n"
        f"Минимальная стратегия (назначения): {min_assignment}, S1 = {S1_min}, S2 = {S2_min}, Потери = {loss_greedy_min}\n"
        f"Максимальная стратегия (назначения): {max_assignment}, S1 = {S1_max}, S2 = {S2_max}, Потери = {loss_greedy_max}\n"
        f"Случайная стратегия (назначения): {random_assignment}, S1 = {S1_random}, S2 = {S2_random}, Потери = {loss_greedy_random}\n"
        f"Венгерский алгоритм (назначения): {hungarian_assignment}, S3 = {S3_hungarian}\n\n"
        f"Матрица C:\n{C}\n\n"
        f"Вектор x:\n{x}\n\n"
        f"Матрица D:\n{D}\n\n"
        f"Матрица G с тильдой:\n{G_tilde}\n\n"
    )

    # Вывод результатов в текстовое поле
    text_output.delete(1.0, tk.END)  # Очищаем текстовое поле перед выводом
    text_output.insert(tk.END, result_text)  # Вставляем текст в текстовое поле

# Создание графического интерфейса
root = tk.Tk()
root.title("Анализ стратегий")
root.geometry("800x950")
root.configure(bg="#1E1E1E")  # Темный фон для корневого окна

# Настройка шрифтов
font_style = tkFont.Font(family="Segoe UI", size=12)  # Используем "Segoe UI"
font_bold = tkFont.Font(family="Segoe UI", size=12, weight="bold")

pad_x = 15
pad_y = 10

btn_color = "#A393EB"  # Цвет кнопок (светло-фиолетовый)
btn_hover_color = "#C6A9FF"  # Цвет кнопок при наведении (яркий сиреневый)
text_bg_color = "#2E2E2E"  # Темный фон текстового поля
text_color = "#FFFFFF"  # Белый текст
frame_bg_color = "#1E1E1E"  # Темный фон фрейма ввода
border_color = "#BBA9FF"  # Цвет рамок и акцентов (мягкий сиреневый)

# Фрейм для ввода данных
input_frame = ttk.Frame(root, padding=(10, 10))
input_frame.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=20, pady=20)
input_frame.configure(style="InputFrame.TFrame")

# Настройка стилей
style = ttk.Style()
style.configure("InputFrame.TFrame", background=frame_bg_color)

# Лейблы и элементы ввода
tk.Label(input_frame, text="Размер матрицы (n):", font=font_style, bg=frame_bg_color, fg=text_color).grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="w")
entry_n = tk.Entry(input_frame, font=font_style, width=10, bd=2, relief="solid", bg=text_bg_color, fg=text_color, highlightbackground=border_color)
entry_n.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky="ew")

tk.Label(input_frame, text="Режим генерации матрицы C:", font=font_style, bg=frame_bg_color, fg=text_color).grid(row=1, column=0, padx=pad_x, pady=pad_y, sticky="w")
matrix_mode = tk.StringVar(value='random')
tk.Radiobutton(input_frame, text="Случайная", variable=matrix_mode, value='random', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky="w")
tk.Radiobutton(input_frame, text="Возрастающая", variable=matrix_mode, value='increasing', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=1, column=2, padx=pad_x, pady=pad_y, sticky="w")
tk.Radiobutton(input_frame, text="Убывающая", variable=matrix_mode, value='decreasing', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=1, column=3, padx=pad_x, pady=pad_y, sticky="w")

tk.Label(input_frame, text="Изменение строк:", font=font_style, bg=frame_bg_color, fg=text_color).grid(row=2, column=0, padx=pad_x, pady=pad_y, sticky="w")
row_mode_var = tk.StringVar(value='random')
tk.Radiobutton(input_frame, text="Возрастающие", variable=row_mode_var, value='increasing', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=2, column=1, padx=pad_x, pady=pad_y, sticky="w")
tk.Radiobutton(input_frame, text="Убывающие", variable=row_mode_var, value='decreasing', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=2, column=2, padx=pad_x, pady=pad_y, sticky="w")
tk.Radiobutton(input_frame, text="Случайные", variable=row_mode_var, value='random', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=2, column=3, padx=pad_x, pady=pad_y, sticky="w")

tk.Label(input_frame, text="Изменение столбцов:", font=font_style, bg=frame_bg_color, fg=text_color).grid(row=3, column=0, padx=pad_x, pady=pad_y, sticky="w")
col_mode_var = tk.StringVar(value='random')
tk.Radiobutton(input_frame, text="Возрастающие", variable=col_mode_var, value='increasing', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky="w")
tk.Radiobutton(input_frame, text="Убывающие", variable=col_mode_var, value='decreasing', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=3, column=2, padx=pad_x, pady=pad_y, sticky="w")
tk.Radiobutton(input_frame, text="Случайные", variable=col_mode_var, value='random', font=font_style, bg=frame_bg_color, fg=text_color).grid(row=3, column=3, padx=pad_x, pady=pad_y, sticky="w")

# Кнопка запуска анализа
button = tk.Button(root, text="Запустить анализ", command=run_analysis, bg=btn_color, fg=text_color, font=font_style, relief="raised", bd=2)
button.grid(row=4, column=0, columnspan=4, pady=pad_y, sticky="ew", padx=20)  # Центрируем кнопку

# Текстовое поле для вывода результатов
text_output = tk.Text(
    root, 
    height=30, 
    width=80, 
    font=font_style, 
    bd=2, 
    relief="sunken", 
    wrap=tk.WORD, 
    bg=text_bg_color,  # Темный фон
    fg=text_color      # Белый текст
)
text_output.grid(row=5, column=0, columnspan=4, padx=20, pady=20, sticky="nsew")

# Добавление полосы прокрутки
scrollbar = tk.Scrollbar(root, command=text_output.yview)
scrollbar.grid(row=5, column=4, sticky='ns')
text_output.config(yscrollcommand=scrollbar.set)

# Настройка растягивания строк и столбцов
for i in range(4):
    root.columnconfigure(i, weight=1)  # Все столбцы растягиваются равномерно
root.rowconfigure(5, weight=1)  # Текстовое поле растягивается по вертикали

root.mainloop()