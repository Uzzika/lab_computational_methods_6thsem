import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QRadioButton, QPushButton, QTextEdit, QButtonGroup, 
    QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QSize
import numpy as np
from scipy.optimize import linear_sum_assignment
from logic import *

class Translator:
    """Class to handle language translations with window-specific texts"""
    def __init__(self):
        self.translations = {
            'ru': {
                'main_window': {
                    'title': "Анализ стратегий",
                    'theme_button': "Переключить на {} тему",
                    'light_theme': "светлую",
                    'dark_theme': "темную",
                    'matrix_size': "Порядок матрицы (n):",
                    'matrix_mode': "Режим генерации матрицы C:",
                    'matrix_modes': ["Возрастающая", "Убывающая", "Случайная"],
                    'row_mode': "Изменение строк:",
                    'col_mode': "Изменение столбцов:",
                    'row_col_modes': ["Возрастающие", "Убывающие", "Случайные"],
                    'run_analysis': "Запустить анализ",
                    'show_matrices': "Показать матрицы",
                    'plot_losses': "Показать график потерь",
                    'strategies': ["Жадная", "Минимальная", "Максимальная", "Случайная"],
                },
                'matrix_window': {
                    'title': "Матрицы и векторы",
                    'matrix_c': "Матрица C:",
                    'vector_x': "Вектор x:",
                    'matrix_d': "Матрица D:",
                    'matrix_g': "Матрица G с тильдой:",
                },
                'messages': {
                    'error_matrix_size': "Введите размер матрицы (n).",
                    'error_matrix_mode': "Выберите режим генерации матрицы C.",
                    'error_row_mode': "Выберите изменение строк.",
                    'error_col_mode': "Выберите изменение столбцов.",
                    'warning_analysis': "Сначала запустите анализ.",
                    'error_title': "Ошибка",
                    'warning_title': "Предупреждение",
                },
                'results': {
                    'title': "Результаты анализа:",
                    'description': "Потери рассчитываются как разница между прибылью, полученной с помощью венгерского алгоритма, и прибылью, полученной с помощью других стратегий.",
                    'strategy': "Стратегия",
                    'assignments': "Назначения",
                    'total_profit': "Общая прибыль (S1)",
                    'updated_profit': "Прибыль от обновленной защиты (S2)",
                    'losses': "Потери (%)",
                },
                'plot': {
                    'title': "Потери стратегий относительно венгерского алгоритма",
                    'strategies': "Стратегии",
                    'losses': "Потери",
                }
            },
            'en': {
                'main_window': {
                    'title': "Strategy Analysis",
                    'theme_button': "Switch to {} theme",
                    'light_theme': "light",
                    'dark_theme': "dark",
                    'matrix_size': "Matrix size (n):",
                    'matrix_mode': "Matrix C generation mode:",
                    'matrix_modes': ["Increasing", "Decreasing", "Random"],
                    'row_mode': "Row change:",
                    'col_mode': "Column change:",
                    'row_col_modes': ["Increasing", "Decreasing", "Random"],
                    'run_analysis': "Run analysis",
                    'show_matrices': "Show matrices",
                    'plot_losses': "Show loss plot",
                    'strategies': ["Greedy", "Minimal", "Maximal", "Random"],
                },
                'matrix_window': {
                    'title': "Matrices and Vectors",
                    'matrix_c': "Matrix C:",
                    'vector_x': "Vector x:",
                    'matrix_d': "Matrix D:",
                    'matrix_g': "Matrix G tilde:",
                },
                'messages': {
                    'error_matrix_size': "Enter matrix size (n).",
                    'error_matrix_mode': "Select matrix C generation mode.",
                    'error_row_mode': "Select row change.",
                    'error_col_mode': "Select column change.",
                    'warning_analysis': "Run analysis first.",
                    'error_title': "Error",
                    'warning_title': "Warning",
                },
                'results': {
                    'title': "Analysis results:",
                    'description': "Losses are calculated as the difference between the profit obtained using the Hungarian algorithm and the profit obtained using other strategies.",
                    'strategy': "Strategy",
                    'assignments': "Assignments",
                    'total_profit': "Total profit (S1)",
                    'updated_profit': "Profit from updated protection (S2)",
                    'losses': "Losses (%)",
                },
                'plot': {
                    'title': "Strategy losses relative to Hungarian algorithm",
                    'strategies': "Strategies",
                    'losses': "Losses",
                }
            }
        }
        self.current_lang = 'ru'
        
    def set_language(self, lang):
        """Set current language"""
        if lang in self.translations:
            self.current_lang = lang
            
    def tr(self, category, key, format_args=None):
        """Get translation for key in specific category"""
        translation = self.translations[self.current_lang][category].get(key, key)
        if format_args is not None:
            if isinstance(format_args, (list, tuple)):
                translation = translation.format(*format_args)
            else:
                translation = translation.format(format_args)
        return translation

translator = Translator()

def _format_matrix(matrix):
    """Format matrix as HTML table"""
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
    """Format vector as HTML table"""
    n = len(vector)
    html = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
    html += "<tr>"
    for i in range(n):
        html += f"<td style='text-align: center;'>{vector[i]:.2f}</td>"
    html += "</tr>"
    html += "</table>"
    return html

class ResponsiveFontMixin:
    """Mixin class to handle responsive font scaling"""
    def __init__(self):
        self.base_font_size = 12
        self.min_font_size = 8
        self.max_font_size = 24
        
    def update_fonts(self, width):
        """Update font sizes based on container width"""
        scale_factor = min(max(width / 800, 0.8), 1.5)
        new_size = max(min(int(self.base_font_size * scale_factor), self.max_font_size), self.min_font_size)
        
        for widget in self.findChildren((QLabel, QPushButton, QRadioButton, QLineEdit, QTextEdit)):
            font = widget.font()
            font.setPointSize(new_size)
            widget.setFont(font)
            
        if hasattr(self, 'text_output'):
            font = self.text_output.font()
            font.setPointSize(new_size)
            self.text_output.setFont(font)

class MatrixWindow(QWidget, ResponsiveFontMixin):
    """Window for displaying matrices and vectors"""
    def __init__(self, matrices_text, dark_theme=True, parent=None):
        super().__init__(parent)
        self.dark_theme = dark_theme
        self.setWindowTitle(translator.tr('matrix_window', 'title'))
        self.setMinimumSize(400, 400)
        
        self.base_font_size = 12
        
        self.text_output = QTextEdit(self)
        self.text_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_output.setReadOnly(True)
        self.text_output.setHtml(matrices_text)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.text_output)
        self.setLayout(layout)

        self.apply_theme()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_fonts(self.width())

    def apply_theme(self):
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

    def update_language(self):
        """Update window title when language changes"""
        self.setWindowTitle(translator.tr('matrix_window', 'title'))

class MainWindow(QMainWindow, ResponsiveFontMixin):
    def __init__(self):
        super().__init__()
        self.dark_theme = True
        self.setWindowTitle(translator.tr('main_window', 'title'))
        self.setMinimumSize(600, 600)
        
        self.base_font_size = 12
        
        # Initialize mode mappings with Russian text (will be updated when language changes)
        self.update_mode_mappings()
        
        self.matrices_text = ""
        self.matrix_window = None
        self.plot_window = None
        self.initUI()
        
    def update_mode_mappings(self):
        """Update mode mappings based on current language"""
        self.matrix_mode_mapping = {
            translator.tr('main_window', 'matrix_modes')[0]: "increasing",
            translator.tr('main_window', 'matrix_modes')[1]: "decreasing",
            translator.tr('main_window', 'matrix_modes')[2]: "random"
        }
        self.row_col_mode_mapping = {
            translator.tr('main_window', 'row_col_modes')[0]: "increasing",
            translator.tr('main_window', 'row_col_modes')[1]: "decreasing",
            translator.tr('main_window', 'row_col_modes')[2]: "random"
        }
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_fonts(self.width())

    def initUI(self):
        central_widget = QWidget()
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Theme and language switcher
        top_buttons_layout = QHBoxLayout()
        
        self.theme_button = QPushButton()
        self.theme_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.theme_button.clicked.connect(self.toggle_theme)
        top_buttons_layout.addWidget(self.theme_button)
        
        self.language_button = QPushButton("EN" if translator.current_lang == 'ru' else "RU")
        self.language_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.language_button.setFixedWidth(50)
        self.language_button.clicked.connect(self.toggle_language)
        top_buttons_layout.addWidget(self.language_button)
        
        main_layout.addLayout(top_buttons_layout)

        # Input frame
        input_frame = QWidget()
        input_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(10)

        # Matrix size (n)
        n_layout = QHBoxLayout()
        n_layout.setSpacing(10)
        self.n_label = QLabel()
        self.n_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.entry_n = QLineEdit()
        self.entry_n.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        n_layout.addWidget(self.n_label)
        n_layout.addWidget(self.entry_n)
        input_layout.addLayout(n_layout)

        # Matrix generation mode
        self.matrix_mode_label = QLabel()
        self.matrix_mode_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        input_layout.addWidget(self.matrix_mode_label)

        self.matrix_mode_group = QButtonGroup()
        matrix_mode_layout = QHBoxLayout()
        matrix_mode_layout.setSpacing(10)
        self.matrix_mode_buttons = []
        for text in translator.tr('main_window', 'matrix_modes'):
            radio = QRadioButton(text)
            radio.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.matrix_mode_group.addButton(radio)
            matrix_mode_layout.addWidget(radio)
            self.matrix_mode_buttons.append(radio)
        input_layout.addLayout(matrix_mode_layout)

        # Row change
        self.row_mode_label = QLabel()
        self.row_mode_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        input_layout.addWidget(self.row_mode_label)

        self.row_mode_group = QButtonGroup()
        row_mode_layout = QHBoxLayout()
        row_mode_layout.setSpacing(10)
        self.row_mode_buttons = []
        for text in translator.tr('main_window', 'row_col_modes'):
            radio = QRadioButton(text)
            radio.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.row_mode_group.addButton(radio)
            row_mode_layout.addWidget(radio)
            self.row_mode_buttons.append(radio)
        input_layout.addLayout(row_mode_layout)

        # Column change
        self.col_mode_label = QLabel()
        self.col_mode_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        input_layout.addWidget(self.col_mode_label)

        self.col_mode_group = QButtonGroup()
        col_mode_layout = QHBoxLayout()
        col_mode_layout.setSpacing(10)
        self.col_mode_buttons = []
        for text in translator.tr('main_window', 'row_col_modes'):
            radio = QRadioButton(text)
            radio.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.col_mode_group.addButton(radio)
            col_mode_layout.addWidget(radio)
            self.col_mode_buttons.append(radio)
        input_layout.addLayout(col_mode_layout)

        main_layout.addWidget(input_frame)

        # Run analysis button
        self.run_button = QPushButton()
        self.run_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.run_button.clicked.connect(self.run_analysis)
        main_layout.addWidget(self.run_button)

        # Show matrices button
        self.show_matrices_button = QPushButton()
        self.show_matrices_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.show_matrices_button.clicked.connect(self.show_matrices)
        main_layout.addWidget(self.show_matrices_button)

        # Plot losses button
        self.plot_button = QPushButton()
        self.plot_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.plot_button.clicked.connect(self.plot_losses)
        main_layout.addWidget(self.plot_button)

        # Results text output
        self.text_output = QTextEdit()
        self.text_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_output.setReadOnly(True)
        main_layout.addWidget(self.text_output)

        self.update_ui_text()
        self.apply_theme()

    def toggle_language(self):
        """Switch between Russian and English"""
        new_lang = 'en' if translator.current_lang == 'ru' else 'ru'
        translator.set_language(new_lang)
        self.language_button.setText("EN" if new_lang == 'ru' else "RU")
        self.update_ui_text()
        self.update_mode_mappings()
        
        # Update child windows if they exist
        if hasattr(self, 'matrix_window') and self.matrix_window:
            self.matrix_window.update_language()
            
        if hasattr(self, 'plot_window') and self.plot_window:
            self.plot_losses()  # Recreate plot with new language
            
        # Update results text if it exists
        if hasattr(self, 'matrices_text') and self.matrices_text:
            self.run_analysis()  # Regenerate results with new language
        
    def update_ui_text(self):
        """Update all UI text elements based on current language"""
        self.setWindowTitle(translator.tr('main_window', 'title'))
        self.theme_button.setText(translator.tr('main_window', 'theme_button', 
            translator.tr('main_window', 'light_theme' if self.dark_theme else 'dark_theme')))
        self.n_label.setText(translator.tr('main_window', 'matrix_size'))
        self.matrix_mode_label.setText(translator.tr('main_window', 'matrix_mode'))
        self.row_mode_label.setText(translator.tr('main_window', 'row_mode'))
        self.col_mode_label.setText(translator.tr('main_window', 'col_mode'))
        
        # Update radio buttons
        for i, text in enumerate(translator.tr('main_window', 'matrix_modes')):
            self.matrix_mode_buttons[i].setText(text)
            
        for i, text in enumerate(translator.tr('main_window', 'row_col_modes')):
            self.row_mode_buttons[i].setText(text)
            self.col_mode_buttons[i].setText(text)
            
        self.run_button.setText(translator.tr('main_window', 'run_analysis'))
        self.show_matrices_button.setText(translator.tr('main_window', 'show_matrices'))
        self.plot_button.setText(translator.tr('main_window', 'plot_losses'))

    def toggle_theme(self):
        self.dark_theme = not self.dark_theme
        self.apply_theme()
        self.theme_button.setText(translator.tr('main_window', 'theme_button', 
            translator.tr('main_window', 'light_theme' if self.dark_theme else 'dark_theme')))

    def apply_theme(self):
        if self.dark_theme:
            self.setStyleSheet("""
                QWidget {
                    background-color: #1E1E1E; 
                    color: #FFFFFF; 
                }
                QMainWindow {
                    border-radius: 15px;
                }
            """)
            self.text_output.setStyleSheet("""
                QTextEdit {
                    background-color: #2E2E2E; 
                    color: #FFFFFF; 
                    border: 1px solid #BBA9FF; 
                    border-radius: 10px;
                    padding: 10px;
                }
            """)
            self.entry_n.setStyleSheet("""
                QLineEdit {
                    background-color: #2E2E2E;
                    color: #FFFFFF; 
                    border: 2px solid #BBA9FF;
                    border-radius: 5px;
                    padding: 5px;
                }
            """)
            button_style = """
                QPushButton {
                    background-color: #A393EB; 
                    color: #FFFFFF; 
                    border: none; 
                    padding: 10px; 
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #8C6FE6;
                }
            """
            self.run_button.setStyleSheet(button_style)
            self.show_matrices_button.setStyleSheet(button_style)
            self.plot_button.setStyleSheet(button_style)
            self.language_button.setStyleSheet(button_style)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #f7fbfc; 
                    color: #000000; 
                }
                QMainWindow {
                    border-radius: 15px;
                }
            """)
            self.text_output.setStyleSheet("""
                QTextEdit {
                    background-color: #ffffff; 
                    color: #000000; 
                    border: 1px solid #769fcd; 
                    border-radius: 10px;
                    padding: 10px;
                }
            """)
            self.entry_n.setStyleSheet("""
                QLineEdit {
                    background-color: #E0E0E0;
                    color: #000000; 
                    border: 2px solid #769fcd;
                    border-radius: 5px;
                    padding: 5px;
                }
            """)
            self.run_button.setStyleSheet("""
                QPushButton {
                    background-color: #769fcd; 
                    color: #ffffff; 
                    border: none; 
                    padding: 10px; 
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #5a8bbd;
                }
            """)
            self.show_matrices_button.setStyleSheet("""
                QPushButton {
                    background-color: #b9d7ea; 
                    color: #000000; 
                    border: none; 
                    padding: 10px; 
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #9fc5e0;
                }
            """)
            self.plot_button.setStyleSheet("""
                QPushButton {
                    background-color: #d6e6f2; 
                    color: #000000; 
                    border: none; 
                    padding: 10px; 
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #c0d8ea;
                }
            """)
            self.language_button.setStyleSheet("""
                QPushButton {
                    background-color: #d6e6f2; 
                    color: #000000; 
                    border: none; 
                    padding: 10px; 
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #c0d8ea;
                }
            """)

    def run_analysis(self):
        try:
            if not self.entry_n.text():
                raise ValueError(translator.tr('messages', 'error_matrix_size'))
            n = int(self.entry_n.text())
            if not self.matrix_mode_group.checkedButton():
                raise ValueError(translator.tr('messages', 'error_matrix_mode'))
            if not self.row_mode_group.checkedButton():
                raise ValueError(translator.tr('messages', 'error_row_mode'))
            if not self.col_mode_group.checkedButton():
                raise ValueError(translator.tr('messages', 'error_col_mode'))
                
            mode = self.matrix_mode_mapping[self.matrix_mode_group.checkedButton().text()]
            row_mode = self.row_col_mode_mapping[self.row_mode_group.checkedButton().text()]
            col_mode = self.row_col_mode_mapping[self.col_mode_group.checkedButton().text()]

            C = generate_matrix(n, mode, row_mode, col_mode)
            chi = generate_x(n)
            D = calculate_D(C, chi)
            G_tilde = calculate_G_tilde(C, chi)

            assert G_tilde.ndim == 2, "Matrix G_tilde should be 2-dimensional"

            greedy_assignment = greedy_strategy(G_tilde)
            hungarian_assignment = hungarian_algorithm(G_tilde)
            min_assignment = min_strategy(D)
            max_assignment = max_strategy(D)
            random_assignment = random_strategy(D)

            S1_greedy = calculate_S1(D, greedy_assignment, chi, C)
            S1_min = calculate_S1(D, min_assignment, chi, C)
            S1_max = calculate_S1(D, max_assignment, chi, C)
            S1_random = calculate_S1(D, random_assignment, chi, C)
            S2_greedy = calculate_S2(G_tilde, greedy_assignment, chi, C)
            S2_min = calculate_S2(G_tilde, min_assignment, chi, C)
            S2_max = calculate_S2(G_tilde, max_assignment, chi, C)
            S2_random = calculate_S2(G_tilde, random_assignment, chi, C)
            S3_hungarian = calculate_S3(G_tilde, hungarian_assignment)

            self.loss_greedy = ((S3_hungarian - S1_greedy) / S3_hungarian) * 100
            self.loss_min = ((S3_hungarian - S1_min) / S3_hungarian) * 100
            self.loss_max = ((S3_hungarian - S1_max) / S3_hungarian) * 100
            self.loss_random = ((S3_hungarian - S1_random) / S3_hungarian) * 100

            self.loss_greedy_min = self.loss_min
            self.loss_greedy_max = self.loss_max
            self.loss_greedy_random = self.loss_random

            result_text = f"""
            <h2 style="color: #BBA9FF;">{translator.tr('results', 'title')}</h2>
            <p style="color: #BBA9FF;">{translator.tr('results', 'description')}</p>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">{translator.tr('results', 'strategy')}</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">{translator.tr('results', 'assignments')}</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">{translator.tr('results', 'total_profit')}</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">{translator.tr('results', 'updated_profit')}</th>
                    <th style="background-color: #2E2E2E; color: #FFFFFF;">{translator.tr('results', 'losses')}</th>
                </tr>
                <tr>
                    <td>{translator.tr('main_window', 'strategies')[0]}</td>
                    <td>{greedy_assignment}</td>
                    <td>{S1_greedy:.2f}</td>
                    <td>{S2_greedy:.2f}</td>
                    <td>{self.loss_greedy:.2f}</td>
                </tr>
                <tr>
                    <td>{translator.tr('main_window', 'strategies')[1]}</td>
                    <td>{min_assignment}</td>
                    <td>{S1_min:.2f}</td>
                    <td>{S2_min:.2f}</td>
                    <td>{self.loss_min:.2f}</td>
                </tr>
                <tr>
                    <td>{translator.tr('main_window', 'strategies')[2]}</td>
                    <td>{max_assignment}</td>
                    <td>{S1_max:.2f}</td>
                    <td>{S2_max:.2f}</td>
                    <td>{self.loss_max:.2f}</td>
                </tr>
                <tr>
                    <td>{translator.tr('main_window', 'strategies')[3]}</td>
                    <td>{random_assignment}</td>
                    <td>{S1_random:.2f}</td>
                    <td>{S2_random:.2f}</td>
                    <td>{self.loss_random:.2f}</td>
                </tr>
            </table>
            """

            self.text_output.setHtml(result_text)

            self.matrices_text = f"""
            <h2 style="color: #BBA9FF;">{translator.tr('matrix_window', 'matrix_c')}</h2>
            {_format_matrix(C)}
            <h2 style="color: #BBA9FF;">{translator.tr('matrix_window', 'vector_x')}</h2>
            {_format_vector(chi)}
            <h2 style="color: #BBA9FF;">{translator.tr('matrix_window', 'matrix_d')}</h2>
            {_format_matrix(D)}
            <h2 style="color: #BBA9FF;">{translator.tr('matrix_window', 'matrix_g')}</h2>
            {_format_matrix(G_tilde)}
            """
        except Exception as e:
            QMessageBox.critical(self, translator.tr('messages', 'error_title'), str(e))

    def show_matrices(self):
        if not self.matrices_text:
            QMessageBox.warning(self, 
                translator.tr('messages', 'warning_title'), 
                translator.tr('messages', 'warning_analysis'))
            return

        self.matrix_window = MatrixWindow(self.matrices_text, self.dark_theme, self)
        self.matrix_window.show()

    def plot_losses(self):
        try:
            if not hasattr(self, 'loss_greedy_min'):
                QMessageBox.warning(self, 
                    translator.tr('messages', 'warning_title'), 
                    translator.tr('messages', 'warning_analysis'))
                return

            strategies = translator.tr('main_window', 'strategies')
            losses = [
                self.loss_greedy_min,
                self.loss_min,
                self.loss_greedy_max,
                self.loss_greedy_random
            ]

            min_loss = min(losses)
            max_loss = max(losses)
            data_range = max_loss - min_loss
            padding = data_range * 0.1

            fig, ax = plt.subplots(figsize=(8, 6))

            if self.dark_theme:
                colors = ['#A393EB', '#BBA9FF', '#8C6FE6', '#6F4FE6']
                bg_color = '#1E1E1E'
                text_color = '#FFFFFF'
            else:
                colors = ['#769fcd', '#b9d7ea', '#d6e6f2', '#a3d2e6']
                bg_color = '#FFFFFF'
                text_color = '#000000'

            bars = ax.bar(strategies, losses, color=colors)
            ax.set_xlabel(translator.tr('plot', 'strategies'), color=text_color)
            ax.set_ylabel(translator.tr('plot', 'losses'), color=text_color)
            ax.set_title(translator.tr('plot', 'title'), color=text_color)

            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)

            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)

            ax.axhline(0, color=text_color, linewidth=0.8)
            ax.set_ylim(min_loss - padding, max_loss + padding)

            for bar in bars:
                height = bar.get_height()
                if height >= 0:
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', color=text_color)
                else:
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, -10),
                                textcoords="offset points",
                                ha='center', va='top', color=text_color)

            self.plot_window = QWidget()
            self.plot_window.setWindowTitle(translator.tr('plot', 'title'))
            self.plot_window.setMinimumSize(600, 500)
            
            layout = QVBoxLayout(self.plot_window)
            layout.setContentsMargins(10, 10, 10, 10)
            
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(canvas)
            
            self.plot_window.show()

        except Exception as e:
            QMessageBox.critical(self, translator.tr('messages', 'error_title'), str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())