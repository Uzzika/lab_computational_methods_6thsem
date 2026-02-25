import numpy as np
from scipy.optimize import linear_sum_assignment

class AssignmentProblemWithConstraints:
    """
    Класс для решения задачи о назначениях с дополнительными ограничениями
    с использованием двойственного алгоритма Удзавы
    """
    
    def __init__(self, C, D_list, b, lambda_init=None, max_iter=100, tol=1e-6):
        """
        Инициализация задачи
        
        Параметры:
        C - матрица стоимостей (n x n)
        D_list - список K матриц ограничений (каждая n x n)
        b - вектор правых частей ограничений (K элементов)
        lambda_init - начальные значения двойственных переменных
        max_iter - максимальное число итераций
        tol - допустимая точность
        """
        self.C = np.array(C)
        self.n = self.C.shape[0]
        self.D_list = [np.array(D) for D in D_list]
        self.K = len(self.D_list)
        self.b = np.array(b)
        self.max_iter = max_iter
        self.tol = tol
        
        # Проверка размерностей
        if len(self.b) != self.K:
            raise ValueError("Количество элементов в b должно совпадать с количеством матриц D")
            
        for D in self.D_list:
            if D.shape != self.C.shape:
                raise ValueError("Все матрицы D должны иметь те же размеры, что и матрица C")
        
        # Инициализация двойственных переменных
        self.lambda_ = np.ones(self.K) if lambda_init is None else np.array(lambda_init)
        
        # История значений для анализа сходимости
        self.history = {
            'lambda': [],
            'constraint_violations': [],
            'objective': []
        }
    
    def solve_assignment(self, G):
        """Решает задачу о назначениях для заданной матрицы G"""
        row_ind, col_ind = linear_sum_assignment(G)
        X = np.zeros_like(G)
        X[row_ind, col_ind] = 1
        return X
    
    def compute_constraint_violations(self, X):
        """Вычисляет нарушение ограничений"""
        violations = []
        for k in range(self.K):
            violation = np.sum(self.D_list[k] * X) - self.b[k]
            violations.append(violation)
        return np.array(violations)
    
    def solve(self, step_size_type='diminishing'):
        """
        Реализация алгоритма Удзавы
        
        Параметры:
        step_size_type - тип шага ('diminishing' - убывающий, 'constant' - постоянный)
        """
        for iteration in range(self.max_iter):
            # Шаг 2: Решаем задачу о назначениях с текущей матрицей G
            G = self.C.copy()
            for k in range(self.K):
                G += self.lambda_[k] * self.D_list[k]
            
            X = self.solve_assignment(G)
            
            # Шаг 3: Вычисляем субградиент (нарушения ограничений)
            violations = self.compute_constraint_violations(X)
            
            # Сохраняем историю
            self.history['lambda'].append(self.lambda_.copy())
            self.history['constraint_violations'].append(violations.copy())
            self.history['objective'].append(np.sum(self.C * X))
            
            # Шаг 4: Проверка критерия остановки
            if np.max(np.abs(violations)) < self.tol:
                print(f"Сходимость достигнута на итерации {iteration}")
                break
                
            # Шаг 5: Обновление двойственных переменных
            if step_size_type == 'diminishing':
                psi = 1.0 / (iteration + 1)  # Убывающий шаг
            elif step_size_type == 'constant':
                psi = 0.1  # Постоянный шаг
            else:
                raise ValueError("Неизвестный тип шага")
                
            self.lambda_ = np.maximum(self.lambda_ + psi * violations, 0)
        
        return X, self.lambda_
    
    def get_history(self):
        """Возвращает историю итераций"""
        return self.history


# Пример использования
if __name__ == "__main__":
    # Пример из пособия: ограничение "не назначать второго работника на вторую работу, 
    # если первому работнику отдана первая работа"
    
    # Матрица стоимостей
    C = np.array([
        [1, 4, 6],
        [7, 2, 5],
        [8, 3, 9]
    ])
    
    # Матрица ограничения
    D1 = np.zeros_like(C)
    D1[0, 0] = 1  # d_11^1 = 1
    D1[1, 1] = 1  # d_22^1 = 1
    D_list = [D1]
    
    # Правая часть ограничения
    b = [1]  # b_1 = 1
    
    # Создаем и решаем задачу
    problem = AssignmentProblemWithConstraints(C, D_list, b, max_iter=100)
    X, lambda_opt = problem.solve(step_size_type='diminishing')
    
    print("Оптимальная матрица назначений:")
    print(X)
    print("Оптимальные двойственные переменные:", lambda_opt)
    print("Значение целевой функции:", np.sum(C * X))
    
    # Проверка ограничений
    violations = problem.compute_constraint_violations(X)
    print("Нарушения ограничений:", violations)