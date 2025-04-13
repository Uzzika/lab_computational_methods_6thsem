import sys
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSpinBox,
                             QLabel, QMessageBox, QGroupBox, QHeaderView, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette


class FirepowerOptimizer:
    def __init__(self, C, k):
        self.C = np.array(C, dtype=int)
        self.k = k
        self.n = len(C)
        self.M = 2 * self.C.max() + 1
        self.computation_time = 0

    def optimize(self):
        if self.n > 6:
            raise ValueError("Размер матрицы слишком велик для данного алгоритма")

        try:
            start_time = time.time()
            sigma_star, S6_sigma_star = self._find_optimal_permutation(self.C)
            sigma_0, S6_sigma_0 = self._find_conjugate_permutation(sigma_star)
            best_solution = self._find_best_solution(sigma_star, S6_sigma_star, sigma_0, S6_sigma_0)
            self.computation_time = time.time() - start_time
            return self._prepare_results(best_solution)
            
        except Exception as e:
            raise RuntimeError(f"Ошибка оптимизации: {str(e)}")

    def _find_optimal_permutation(self, matrix):
        row_ind, col_ind = linear_sum_assignment(-matrix)
        sigma = col_ind.tolist()
        S = sum(matrix[sigma[j], j] for j in range(self.n))
        return sigma, S

    def _find_conjugate_permutation(self, sigma):
        G = np.where(np.arange(self.n)[:, None] != np.array(sigma)[None, :],
                    self.C + self.M,
                    0)
        row_ind, col_ind = linear_sum_assignment(-G)
        conjugate_sigma = col_ind.tolist()
        S = sum(self.C[conjugate_sigma[j], j] for j in range(self.n))
        return conjugate_sigma, S

    def _find_complementary_permutation(self, sigma, gamma_vector):
        G = np.where(np.arange(self.n)[:, None] != np.array(sigma)[None, :],
                    self.C + self.M,
                    np.where(np.array(gamma_vector)[None, :],
                            self.C + 2*self.M,
                            0))
        
        row_ind, col_ind = linear_sum_assignment(-G)
        sigma_comp = col_ind.tolist()
        S = sum(self.C[sigma_comp[j], j] for j in range(self.n))
        return sigma_comp, S

    def _find_best_solution(self, sigma_star, S_star, sigma_0, S0):
        best = {'sigmas': (sigma_star, sigma_0), 'S': S_star + S0}
        
        for gamma in range(1, 2**self.n - 1):
            gamma_vector = [int(b) for b in f"{gamma:0{self.n}b}"]
            sigma_1, S1 = self._find_complementary_permutation(sigma_star, gamma_vector)
            sigma_2, S2 = self._find_conjugate_permutation(sigma_1)
            
            if S1 + S2 > best['S']:
                best = {'sigmas': (sigma_1, sigma_2), 'S': S1 + S2}
        
        return best

    def _prepare_results(self, solution):
        sigma1, sigma2 = solution['sigmas']
        schedule = [[sigma1[j], sigma2[j]] for j in range(self.n)]
        total_power = self.C.sum() - (self.k - 1) / self.k * solution['S']
        
        return {
            'schedule': schedule,
            'total_power': total_power,
            'sigma1': sigma1,
            'sigma2': sigma2,
            'S7': solution['S'],
            'initial_power': self.C.sum(),
            'computation_time': self.computation_time
        }


class MatrixEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Оптимизатор боевой мощи")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 1em;
                font-weight: bold;
                background-color: white;
            }
            QGroupBox::title {
                color: #2980b9;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QTableWidget {
                border: 1px solid #bdc3c7;
                gridline-color: #ecf0f1;
                border-radius: 4px;
            }
            QSpinBox {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
            }
            QLabel {
                color: #2c3e50;
            }
        """)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок
        title = QLabel("Оптимизация распределения боевой мощи")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title)

        # Параметры
        params_group = QGroupBox("Параметры")
        params_layout = QHBoxLayout()

        size_layout = QHBoxLayout()
        size_label = QLabel("Размер матрицы:")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(2, 6)
        self.size_spin.setValue(3)
        self.size_spin.valueChanged.connect(self.resize_matrix)
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_spin)

        k_layout = QHBoxLayout()
        k_label = QLabel("Коэффициент k:")
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 10)
        self.k_spin.setValue(2)
        k_layout.addWidget(k_label)
        k_layout.addWidget(self.k_spin)

        params_layout.addLayout(size_layout)
        params_layout.addStretch()
        params_layout.addLayout(k_layout)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Матрица
        matrix_group = QGroupBox("Матрица мощности")
        matrix_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget::item {
                padding: 5px;
                text-align: center;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 5px;
                border: none;
            }
        """)
        matrix_layout.addWidget(self.table)
        
        buttons_layout = QHBoxLayout()
        self.example_btn = QPushButton("Загрузить пример")
        self.clear_btn = QPushButton("Очистить")
        self.calc_btn = QPushButton("Рассчитать")
        self.calc_btn.setStyleSheet("background-color: #27ae60;")
        
        buttons_layout.addWidget(self.example_btn)
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addWidget(self.calc_btn)
        
        matrix_layout.addLayout(buttons_layout)
        matrix_group.setLayout(matrix_layout)
        layout.addWidget(matrix_group)

        # Результаты
        results_group = QGroupBox("Результаты")
        results_layout = QVBoxLayout()
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        results_layout.addWidget(self.result_display)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Подключение функций
        self.example_btn.clicked.connect(self.load_example)
        self.clear_btn.clicked.connect(self.init_matrix)
        self.calc_btn.clicked.connect(self.calculate)

        self.init_matrix()

    def init_matrix(self):
        n = self.size_spin.value()
        self.table.setRowCount(n)
        self.table.setColumnCount(n)
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem("1")
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)

    def resize_matrix(self):
        self.init_matrix()

    def get_matrix(self):
        n = self.size_spin.value()
        C = []
        for i in range(n):
            row = []
            for j in range(n):
                try:
                    val = int(self.table.item(i, j).text())
                    if val < 0:
                        raise ValueError("Значения должны быть положительными")
                    row.append(val)
                except Exception:
                    QMessageBox.warning(self, "Ошибка", 
                        f"Некорректное значение в ячейке [{i+1}, {j+1}]")
                    raise
            C.append(row)
        return C

    def load_example(self):
        example = [[5, 4, 2], [4, 5, 4], [2, 4, 5]]
        self.size_spin.setValue(3)
        self.resize_matrix()
        for i in range(3):
            for j in range(3):
                self.table.item(i, j).setText(str(example[i][j]))

    def calculate(self):
        try:
            C = self.get_matrix()
            k = self.k_spin.value()
            optimizer = FirepowerOptimizer(C, k)
            results = optimizer.optimize()
            
            self.result_display.setHtml(f"""
                <div style='font-family: Arial; padding: 10px;'>
                    <h3 style='color: #2c3e50;'>Результаты оптимизации:</h3>
                    <p><b>Исходная мощность:</b> {results['initial_power']}</p>
                    <p><b>Итоговая мощность:</b> {results['total_power']:.2f}</p>
                    <p><b>Эффективность:</b> {(1 - results['total_power']/results['initial_power'])*100:.1f}%</p>
                    <p><b>Время расчета:</b> {results['computation_time']:.3f} сек</p>
                    <h4 style='color: #2c3e50;'>Оптимальное расписание:</h4>
                    {self._format_schedule(results['schedule'])}
                </div>
            """)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def _format_schedule(self, schedule):
        return ''.join(
            f"<p>Период {j+1}: <b>Отряды {t1+1} и {t2+1}</b></p>"
            for j, (t1, t2) in enumerate(schedule)
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MatrixEditor()
    window.show()
    sys.exit(app.exec_())