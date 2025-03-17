import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QRadioButton, QPushButton, QTextEdit, QButtonGroup, QMessageBox
)
from PyQt5.QtGui import QFont
import numpy as np
from scipy.optimize import linear_sum_assignment
from logic import *

def _format_matrix(matrix):
    """Форматирует матрицу в виде HTML-таблицы."""
    n = len(matrix)
    html = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
    for i in range(n):
        html += "<tr>"
        for j in range(n):
            html += f"<td style='text-align: center;'>{matrix[i, j]:.2f}</td>"
        html += "</tr>"
    html += "</table>"
    return html

def _format_vector(vector):
    """Форматирует вектор в виде HTML-таблицы."""
    n = len(vector)
    html = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
    html += "<tr>"
    for i in range(n):
        html += f"<td style='text-align: center;'>{vector[i]:.2f}</td>"
    html += "</tr>"
    html += "</table>"
    return html

class MatrixWindow(QWidget):
    """Окно для отображения матриц и векторов."""
    def __init__(self, matrices_text, dark_theme=True):
        super().__init__()
        self.dark_theme = dark_theme
        self.setWindowTitle("Матрицы и векторы")
        self.setGeometry(200, 200, 800, 800)

        # Текстовое поле для вывода матриц
        self.text_output = QTextEdit(self)
        self.text_output.setFont(QFont("Segoe UI", 12))
        self.text_output.setReadOnly(True)
        self.text_output.setHtml(matrices_text)

        # Основной макет
        layout = QVBoxLayout(self)
        layout.addWidget(self.text_output)
        self.setLayout(layout)

        # Применяем тему после создания всех элементов
        self.apply_theme()

    def apply_theme(self):
        """Применяет тему к окну."""
        if self.dark_theme:
            self.setStyleSheet("""
                background-color: #1E1E1E; 
                color: #FFFFFF; 
                border-radius: 15px;
            """)
            self.text_output.setStyleSheet("""
                background-color: #2E2E2E; 
                color: #FFFFFF; 
                border: 1px solid #BBA9FF; 
                border-radius: 10px;
                padding: 10px;
            """)
        else:
            self.setStyleSheet("""
                background-color: #f7fbfc; 
                color: #000000; 
                border-radius: 15px;
            """)
            self.text_output.setStyleSheet("""
                background-color: #ffffff; 
                color: #000000; 
                border: 1px solid #769fcd; 
                border-radius: 10px;
                padding: 10px;
            """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dark_theme = True  # По умолчанию тема темная
        self.setWindowTitle("Анализ стратегий")
        self.setGeometry(100, 100, 800, 800)

        # Словари для перевода текста кнопок в режимы
        self.matrix_mode_mapping = {
            "Возрастающая": "increasing",
            "Убывающая": "decreasing",
            "Случайная": "random"
        }
        self.row_col_mode_mapping = {
            "Возрастающие": "increasing",
            "Убывающие": "decreasing",
            "Случайные": "random"
        }

        # Переменная для хранения текста с матрицами
        self.matrices_text = ""

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Кнопка для переключения темы
        self.theme_button = QPushButton("Переключить на светлую тему" if self.dark_theme else "Переключить на темную тему")
        self.theme_button.setFont(QFont("Segoe UI", 12))
        self.theme_button.clicked.connect(self.toggle_theme)
        main_layout.addWidget(self.theme_button)

        # Фрейм для ввода данных
        input_frame = QWidget()
        input_layout = QVBoxLayout(input_frame)

        # Размер матрицы (n)
        n_layout = QHBoxLayout()
        n_label = QLabel("Размер матрицы (n):")
        n_label.setFont(QFont("Segoe UI", 12))
        self.entry_n = QLineEdit()
        self.entry_n.setFont(QFont("Segoe UI", 12))
        n_layout.addWidget(n_label)
        n_layout.addWidget(self.entry_n)
        input_layout.addLayout(n_layout)

        # Режим генерации матрицы C
        matrix_mode_label = QLabel("Режим генерации матрицы C:")
        matrix_mode_label.setFont(QFont("Segoe UI", 12))
        input_layout.addWidget(matrix_mode_label)

        self.matrix_mode_group = QButtonGroup()
        matrix_mode_layout = QHBoxLayout()
        for text in ["Возрастающая", "Убывающая", "Случайная"]:
            radio = QRadioButton(text)
            radio.setFont(QFont("Segoe UI", 12))
            self.matrix_mode_group.addButton(radio)
            matrix_mode_layout.addWidget(radio)
        input_layout.addLayout(matrix_mode_layout)

        # Изменение строк
        row_mode_label = QLabel("Изменение строк:")
        row_mode_label.setFont(QFont("Segoe UI", 12))
        input_layout.addWidget(row_mode_label)

        self.row_mode_group = QButtonGroup()
        row_mode_layout = QHBoxLayout()
        for text in ["Возрастающие", "Убывающие", "Случайные"]:
            radio = QRadioButton(text)
            radio.setFont(QFont("Segoe UI", 12))
            self.row_mode_group.addButton(radio)
            row_mode_layout.addWidget(radio)
        input_layout.addLayout(row_mode_layout)

        # Изменение столбцов
        col_mode_label = QLabel("Изменение столбцов:")
        col_mode_label.setFont(QFont("Segoe UI", 12))
        input_layout.addWidget(col_mode_label)

        self.col_mode_group = QButtonGroup()
        col_mode_layout = QHBoxLayout()
        for text in ["Возрастающие", "Убывающие", "Случайные"]:
            radio = QRadioButton(text)
            radio.setFont(QFont("Segoe UI", 12))
            self.col_mode_group.addButton(radio)
            col_mode_layout.addWidget(radio)
        input_layout.addLayout(col_mode_layout)

        main_layout.addWidget(input_frame)

        # Кнопка запуска анализа
        self.run_button = QPushButton("Запустить анализ")
        self.run_button.setFont(QFont("Segoe UI", 12))
        self.run_button.clicked.connect(self.run_analysis)
        main_layout.addWidget(self.run_button)

        # Кнопка для отображения матриц
        self.show_matrices_button = QPushButton("Показать матрицы")
        self.show_matrices_button.setFont(QFont("Segoe UI", 12))
        self.show_matrices_button.clicked.connect(self.show_matrices)
        main_layout.addWidget(self.show_matrices_button)

        # Кнопка для вывода графика потерь
        self.plot_button = QPushButton("Показать график потерь")
        self.plot_button.setFont(QFont("Segoe UI", 12))
        self.plot_button.clicked.connect(self.plot_losses)
        main_layout.addWidget(self.plot_button)

        # Текстовое поле для вывода результатов
        self.text_output = QTextEdit()
        self.text_output.setFont(QFont("Segoe UI", 12))
        self.text_output.setReadOnly(True)
        main_layout.addWidget(self.text_output)

        # Применяем тему после создания всех элементов
        self.apply_theme()

    def toggle_theme(self):
        """Переключает тему между темной и светлой."""
        self.dark_theme = not self.dark_theme
        self.apply_theme()
        self.theme_button.setText("Переключить на светлую тему" if self.dark_theme else "Переключить на темную тему")

    def apply_theme(self):
        """Применяет текущую тему."""
        if self.dark_theme:
            # Темная тема
            self.setStyleSheet("""
                background-color: #1E1E1E; 
                color: #FFFFFF; 
                border-radius: 15px;
            """)
            self.text_output.setStyleSheet("""
                background-color: #2E2E2E; 
                color: #FFFFFF; 
                border: 1px solid #BBA9FF; 
                border-radius: 10px;
                padding: 10px;
            """)
            self.entry_n.setStyleSheet("""
                background-color: #2E2E2E;  /* Чуть светлее для темной темы */
                color: #FFFFFF; 
                border: 2px solid #BBA9FF;  /* Увеличиваем толщину границы */
                border-radius: 5px;
                padding: 5px;
            """)
            self.run_button.setStyleSheet("""
                background-color: #A393EB; 
                color: #FFFFFF; 
                border: none; 
                padding: 10px; 
                border-radius: 10px;
            """)
            self.show_matrices_button.setStyleSheet("""
                background-color: #A393EB; 
                color: #FFFFFF; 
                border: none; 
                padding: 10px; 
                border-radius: 10px;
            """)
            self.plot_button.setStyleSheet("""
                background-color: #A393EB; 
                color: #FFFFFF; 
                border: none; 
                padding: 10px; 
                border-radius: 10px;
            """)
        else:
            # Светлая тема
            self.setStyleSheet("""
                background-color: #f7fbfc; 
                color: #000000; 
                border-radius: 15px;
            """)
            self.text_output.setStyleSheet("""
                background-color: #ffffff; 
                color: #000000; 
                border: 1px solid #769fcd; 
                border-radius: 10px;
                padding: 10px;
            """)
            self.entry_n.setStyleSheet("""
                background-color: #E0E0E0;  /* Чуть темнее для светлой темы */
                color: #000000; 
                border: 2px solid #769fcd;  /* Увеличиваем толщину границы */
                border-radius: 5px;
                padding: 5px;
            """)
            self.run_button.setStyleSheet("""
                background-color: #769fcd; 
                color: #ffffff; 
                border: none; 
                padding: 10px; 
                border-radius: 10px;
            """)
            self.show_matrices_button.setStyleSheet("""
                background-color: #b9d7ea; 
                color: #000000; 
                border: none; 
                padding: 10px; 
                border-radius: 10px;
            """)
            self.plot_button.setStyleSheet("""
                background-color: #d6e6f2; 
                color: #000000; 
                border: none; 
                padding: 10px; 
                border-radius: 10px;
            """)

    def run_analysis(self):
        """Запуск анализа."""
        try:
            # Проверка, что размер матрицы введен
            if not self.entry_n.text():
                raise ValueError("Введите размер матрицы (n).")

            n = int(self.entry_n.text())

            # Проверка, что выбран режим генерации матрицы C
            if not self.matrix_mode_group.checkedButton():
                raise ValueError("Выберите режим генерации матрицы C.")

            # Проверка, что выбрано изменение строк
            if not self.row_mode_group.checkedButton():
                raise ValueError("Выберите изменение строк.")

            # Проверка, что выбрано изменение столбцов
            if not self.col_mode_group.checkedButton():
                raise ValueError("Выберите изменение столбцов.")

            # Получение режимов
            mode = self.matrix_mode_mapping[self.matrix_mode_group.checkedButton().text()]
            row_mode = self.row_col_mode_mapping[self.row_mode_group.checkedButton().text()]
            col_mode = self.row_col_mode_mapping[self.col_mode_group.checkedButton().text()]

            # Генерация матрицы C и вектора chi
            C = generate_matrix(n, mode, row_mode, col_mode)  # Убедитесь, что C — двумерный массив
            chi = generate_x(n)  # Убедитесь, что chi — одномерный массив

            # Вычисление матрицы D
            D = calculate_D(C, chi)  # Убедитесь, что D — двумерный массив

            # Вычисление матрицы G с тильдой
            G_tilde = calculate_G_tilde(C, chi)  # Убедитесь, что G_tilde — двумерный массив

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

            S2_greedy = calculate_S2(D, greedy_assignment, chi, C)
            S2_min = calculate_S2(D, min_assignment, chi, C)
            S2_max = calculate_S2(D, max_assignment, chi, C)
            S2_random = calculate_S2(D, random_assignment, chi, C)

            S3_hungarian = calculate_S3(G_tilde, hungarian_assignment)

            # Потери
            self.loss_greedy = S3_hungarian - S1_greedy
            self.loss_min = S3_hungarian - S1_min
            self.loss_max = S3_hungarian - S1_max
            self.loss_random = S3_hungarian - S1_random

            # Инициализация атрибутов для графика
            self.loss_greedy_min = self.loss_greedy
            self.loss_greedy_max = self.loss_max
            self.loss_greedy_random = self.loss_random

            # Формирование строки для вывода с использованием HTML
            result_text = """
            <h2 style="color: #BBA9FF;">Результаты анализа:</h2>
            <p style="color: #BBA9FF;">Потери рассчитываются как разница между прибылью, полученной с помощью венгерского алгоритма, и прибылью, полученной с помощью других стратегий.</p>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">Стратегия</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">Назначения</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">Общая прибыль (S1)</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">Прибыль от обновленной защиты (S2)</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">Потери</th>
                </tr>
                <tr>
                    <td>Жадная</td>
                    <td>{}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                </tr>
                <tr>
                    <td>Минимальная</td>
                    <td>{}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                </tr>
                <tr>
                    <td>Максимальная</td>
                    <td>{}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                </tr>
                <tr>
                    <td>Случайная</td>
                    <td>{}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                </tr>
                <tr>
                    <td>Венгерский алгоритм</td>
                    <td>{}</td>
                    <td>-</td>
                    <td>-</td>
                    <td>{:.2f}</td>
                </tr>
            </table>
            """.format(
                greedy_assignment, S1_greedy, S2_greedy, self.loss_greedy,
                min_assignment, S1_min, S2_min, self.loss_min,
                max_assignment, S1_max, S2_max, self.loss_max,
                random_assignment, S1_random, S2_random, self.loss_random,
                hungarian_assignment, S3_hungarian
            )

            # Вывод результатов в текстовое поле
            self.text_output.setHtml(result_text)

            # Формирование HTML-строки для вывода матриц и векторов
            self.matrices_text = """
            <h2 style="color: #BBA9FF;">Матрица C:</h2>
            {}
            <h2 style="color: #BBA9FF;">Вектор x:</h2>
            {}
            <h2 style="color: #BBA9FF;">Матрица D:</h2>
            {}
            <h2 style="color: #BBA9FF;">Матрица G с тильдой:</h2>
            {}
            """.format(
                _format_matrix(C),  # Форматируем матрицу C
                _format_vector(chi),  # Форматируем вектор chi
                _format_matrix(D),  # Форматируем матрицу D
                _format_matrix(G_tilde)  # Форматируем матрицу G_tilde
            )

        except Exception as e:
            # Вывод сообщения об ошибке
            QMessageBox.critical(self, "Ошибка", str(e))

    def show_matrices(self):
        """Открывает окно с матрицами."""
        if not self.matrices_text:
            QMessageBox.warning(self, "Предупреждение", "Сначала запустите анализ.")
            return

        # Создание и отображение окна с матрицами
        self.matrix_window = MatrixWindow(self.matrices_text, self.dark_theme)
        self.matrix_window.show()

    def plot_losses(self):
        """Строит график потерь для всех стратегий."""
        try:
            # Проверяем, что анализ был запущен
            if not hasattr(self, 'loss_greedy_min'):
                QMessageBox.warning(self, "Ошибка", "Сначала запустите анализ.")
                return

            # Данные для графика
            strategies = ["Жадная", "Минимальная", "Максимальная", "Случайная"]
            losses = [
                self.loss_greedy_min,  # Потери для жадной стратегии
                self.loss_min,         # Потери для минимальной стратегии
                self.loss_greedy_max,  # Потери для максимальной стратегии
                self.loss_greedy_random  # Потери для случайной стратегии
            ]

            # Вычисляем диапазон данных
            min_loss = min(losses)
            max_loss = max(losses)
            data_range = max_loss - min_loss

            # Динамически вычисляем отступы (padding)
            padding = data_range * 0.1  # 10% от диапазона данных

            # Создаем график
            fig, ax = plt.subplots()

            # Цвета для светлой и темной темы
            if self.dark_theme:
                colors = ['#A393EB', '#BBA9FF', '#8C6FE6', '#6F4FE6']  # Фиолетовые оттенки для темной темы
                bg_color = '#1E1E1E'  # Цвет фона для темной темы
                text_color = '#FFFFFF'  # Цвет текста для темной темы
            else:
                colors = ['#769fcd', '#b9d7ea', '#d6e6f2', '#a3d2e6']  # Голубые оттенки для светлой темы
                bg_color = '#FFFFFF'  # Цвет фона для светлой темы
                text_color = '#000000'  # Цвет текста для светлой темы

            # Применяем цвета к графику
            bars = ax.bar(strategies, losses, color=colors)
            ax.set_xlabel("Стратегии", color=text_color)
            ax.set_ylabel("Потери", color=text_color)
            ax.set_title("Потери стратегий относительно венгерского алгоритма", color=text_color)

            # Устанавливаем цвет фона графика
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)

            # Настройка цветов осей и текста
            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)

            # Добавляем ось X в середине графика
            ax.axhline(0, color=text_color, linewidth=0.8)  # Ось X

            # Устанавливаем динамические границы оси Y
            ax.set_ylim(min_loss - padding, max_loss + padding)

            # Добавляем значения потерь над/под столбцами
            for bar in bars:
                height = bar.get_height()
                if height >= 0:
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # Смещение текста относительно столбца
                                textcoords="offset points",
                                ha='center', va='bottom', color=text_color)
                else:
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, -10),  # Смещение текста относительно столбца
                                textcoords="offset points",
                                ha='center', va='top', color=text_color)

            # Отображаем график в новом окне
            self.plot_window = QWidget()
            self.plot_window.setWindowTitle("График потерь")
            self.plot_window.setGeometry(100, 100, 800, 600)
            layout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.plot_window.setLayout(layout)
            self.plot_window.show()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())