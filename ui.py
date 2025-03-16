import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QRadioButton, QPushButton, QTextEdit, QButtonGroup, QMessageBox
)
from PyQt5.QtGui import QFont
from logic import *

def _format_matrix(self, matrix):
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

def _format_vector(self, vector):
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
    def __init__(self, matrices_text):
        super().__init__()
        self.setWindowTitle("Матрицы и векторы")
        self.setGeometry(200, 200, 800, 800)  # Увеличили размер окна для удобства
        self.setStyleSheet("""
            background-color: #1E1E1E; 
            color: #FFFFFF; 
            border-radius: 15px;
        """)

        # Текстовое поле для вывода матриц
        self.text_output = QTextEdit(self)
        self.text_output.setFont(QFont("Segoe UI", 12))
        self.text_output.setStyleSheet("""
            background-color: #2E2E2E; 
            color: #FFFFFF; 
            border: 1px solid #BBA9FF; 
            border-radius: 10px;
            padding: 10px;
        """)
        self.text_output.setReadOnly(True)
        self.text_output.setHtml(matrices_text)  # Используем HTML для форматирования

        # Основной макет
        layout = QVBoxLayout(self)
        layout.addWidget(self.text_output)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ стратегий")
        self.setGeometry(100, 100, 800, 800)  # Уменьшили высоту окна
        self.setStyleSheet("""
            background-color: #1E1E1E; 
            color: #FFFFFF; 
            border-radius: 15px;
        """)

        # Словари для перевода текста кнопок в режимы
        self.matrix_mode_mapping = {
            "Случайная": "random",
            "Возрастающая": "increasing",
            "Убывающая": "decreasing"
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

        # Фрейм для ввода данных
        input_frame = QWidget()
        input_layout = QVBoxLayout(input_frame)

        # Размер матрицы (n)
        n_layout = QHBoxLayout()
        n_label = QLabel("Размер матрицы (n):")
        n_label.setFont(QFont("Segoe UI", 12))
        self.entry_n = QLineEdit()
        self.entry_n.setFont(QFont("Segoe UI", 12))
        self.entry_n.setStyleSheet("""
            background-color: #2E2E2E; 
            color: #FFFFFF; 
            border: 1px solid #BBA9FF; 
            border-radius: 10px;
            padding: 5px;
        """)
        n_layout.addWidget(n_label)
        n_layout.addWidget(self.entry_n)
        input_layout.addLayout(n_layout)

        # Режим генерации матрицы C
        matrix_mode_label = QLabel("Режим генерации матрицы C:")
        matrix_mode_label.setFont(QFont("Segoe UI", 12))
        input_layout.addWidget(matrix_mode_label)

        self.matrix_mode_group = QButtonGroup()
        matrix_mode_layout = QHBoxLayout()
        for text in ["Случайная", "Возрастающая", "Убывающая"]:
            radio = QRadioButton(text)
            radio.setFont(QFont("Segoe UI", 12))
            radio.setStyleSheet("color: #FFFFFF;")
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
            radio.setStyleSheet("color: #FFFFFF;")
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
            radio.setStyleSheet("color: #FFFFFF;")
            self.col_mode_group.addButton(radio)
            col_mode_layout.addWidget(radio)
        input_layout.addLayout(col_mode_layout)

        main_layout.addWidget(input_frame)

        # Кнопка запуска анализа
        self.run_button = QPushButton("Запустить анализ")
        self.run_button.setFont(QFont("Segoe UI", 12))
        self.run_button.setStyleSheet("""
            background-color: #A393EB; 
            color: #FFFFFF; 
            border: none; 
            padding: 10px; 
            border-radius: 10px;
        """)
        self.run_button.clicked.connect(self.run_analysis)
        main_layout.addWidget(self.run_button)

        # Кнопка для отображения матриц
        self.show_matrices_button = QPushButton("Показать матрицы")
        self.show_matrices_button.setFont(QFont("Segoe UI", 12))
        self.show_matrices_button.setStyleSheet("""
            background-color: #A393EB; 
            color: #FFFFFF; 
            border: none; 
            padding: 10px; 
            border-radius: 10px;
        """)
        self.show_matrices_button.clicked.connect(self.show_matrices)
        main_layout.addWidget(self.show_matrices_button)

        # Текстовое поле для вывода результатов
        self.text_output = QTextEdit()
        self.text_output.setFont(QFont("Segoe UI", 12))
        self.text_output.setStyleSheet("""
            background-color: #2E2E2E; 
            color: #FFFFFF; 
            border: 1px solid #BBA9FF; 
            border-radius: 10px;
            padding: 10px;
        """)
        self.text_output.setReadOnly(True)
        main_layout.addWidget(self.text_output)

    def _format_matrix(self, matrix):
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

    def _format_vector(self, vector):
        """Форматирует вектор в виде HTML-таблицы."""
        n = len(vector)
        html = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
        html += "<tr>"
        for i in range(n):
            html += f"<td style='text-align: center;'>{vector[i]:.2f}</td>"
        html += "</tr>"
        html += "</table>"
        return html

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

            # Генерация матрицы C и вектора x
            C = generate_matrix(n, mode, row_mode, col_mode)
            x = generate_x(n)

            # Вычисление матрицы D
            D = calculate_D(C, x, x)

            # Вычисление матрицы G с тильдой
            G_tilde = calculate_G_tilde(C, x, x)

            # Применение стратегий
            greedy_assignment = greedy_strategy(D)
            hungarian_assignment = hungarian_algorithm(G_tilde)
            min_assignment = min_strategy(D)
            max_assignment = max_strategy(D)
            random_assignment = random_strategy(D)

            # Вычисление целевых функций
            S1_greedy = calculate_S1(D, greedy_assignment, x, C, x)
            S1_min = calculate_S1(D, min_assignment, x, C, x)
            S1_max = calculate_S1(D, max_assignment, x, C, x)
            S1_random = calculate_S1(D, random_assignment, x, C, x)

            S2_greedy = calculate_S2(D, greedy_assignment, x, C, x)
            S2_min = calculate_S2(D, min_assignment, x, C, x)
            S2_max = calculate_S2(D, max_assignment, x, C, x)
            S2_random = calculate_S2(D, random_assignment, x, C, x)

            S3_hungarian = calculate_S3(G_tilde, hungarian_assignment)

            # Оценка проигрыша жадной стратегии
            loss_greedy_min = S3_hungarian - S1_min
            loss_greedy_max = S3_hungarian - S1_max
            loss_greedy_random = S3_hungarian - S1_random

            # Формирование строки для вывода с использованием HTML
            result_text = """
            <h2 style="color: #BBA9FF;">Результаты анализа:</h2>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">Стратегия</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">Назначения</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">S1</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">S2</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">Потери</th>
                </tr>
                <tr>
                    <td>Жадная</td>
                    <td>{}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>-</td>
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
                greedy_assignment, S1_greedy, S2_greedy,
                min_assignment, S1_min, S2_min, loss_greedy_min,
                max_assignment, S1_max, S2_max, loss_greedy_max,
                random_assignment, S1_random, S2_random, loss_greedy_random,
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
                self._format_matrix(C),  # Форматируем матрицу C
                self._format_vector(x),  # Форматируем вектор x
                self._format_matrix(D),  # Форматируем матрицу D
                self._format_matrix(G_tilde)  # Форматируем матрицу G_tilde
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
        self.matrix_window = MatrixWindow(self.matrices_text)
        self.matrix_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())