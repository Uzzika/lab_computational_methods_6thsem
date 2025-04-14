import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import json

class FirepowerOptimizer:
    def __init__(self, C, k):
        # Проверка входных данных
        if not isinstance(C, list) or not all(isinstance(row, list) for row in C):
            raise ValueError("Матрица должна быть списком списков")
        if len(C) == 0 or any(len(row) != len(C) for row in C):
            raise ValueError("Матрица должна быть квадратной")
        if k <= 1:
            raise ValueError("Коэффициент k должен быть больше 1")

        self.C = np.array(C, dtype=int)
        self.k = float(k)
        self.n = len(C)
        self.M = 2 * self.C.max() + 1  # Большая константа
        self.computation_time = 0

    def optimize(self):
        try:
            start_time = time.time()
            
            if (self.C < 0).any():
                raise ValueError("Матрица содержит отрицательные значения")

            if self.n == 2:
                return self._solve_2x2_case()
            else:
                return self._optimize_task3()
                
        except Exception as e:
            raise RuntimeError(f"Ошибка оптимизации: {str(e)}")
        finally:
            self.computation_time = time.time() - start_time

    def _solve_2x2_case(self):
        """Специальное решение для матрицы 2x2"""
        options = [
            [[0, 1], [0, 1]],  # Оба стреляют по одним и тем же отрядам
            [[0, 1], [1, 0]],   # Перекрестное расписание
        ]
        
        best_power = float('inf')
        best_schedule = None
        
        for schedule in options:
            total = 0
            for j in range(2):
                for i in schedule[j]:
                    total += self.C[i, j] / (self.k if j > 0 and i in schedule[j-1] else 1)
            
            if total < best_power:
                best_power = total
                best_schedule = schedule
        
        return {
            'schedule': best_schedule,
            'total_power': float(best_power),
            'initial_power': int(self.C.sum()),
            'computation_time': self.computation_time
        }

    def _optimize_task3(self):
        """Оптимизация для матриц 3x3 и больше по алгоритму из пособия"""
        # 1. Находим оптимальную перестановку σ*
        sigma_star, S6_sigma_star = self._find_optimal_permutation(self.C)
        
        # 2. Находим сопряженную перестановку σ0
        sigma_0, S6_sigma_0 = self._find_conjugate_permutation(sigma_star)
        
        # 3. Ищем лучшее решение
        best_solution = self._find_best_solution(sigma_star, S6_sigma_star, sigma_0, S6_sigma_0)
        
        # 4. Формируем результаты
        sigma1, sigma2 = best_solution['sigmas']
        schedule = [[int(sigma1[j]), int(sigma2[j])] for j in range(self.n)]
        total_power = float(self.C.sum() - (self.k - 1) / self.k * best_solution['S'])
        
        return {
            'schedule': schedule,
            'total_power': total_power,
            'initial_power': int(self.C.sum()),
            'computation_time': self.computation_time
        }

    def _find_optimal_permutation(self, matrix):
        """Нахождение оптимальной перестановки σ* (задача о назначениях)"""
        row_ind, col_ind = linear_sum_assignment(-matrix)
        sigma = col_ind[np.argsort(row_ind)].tolist()
        S = sum(matrix[sigma[j], j] for j in range(self.n))
        return sigma, S

    def _find_conjugate_permutation(self, sigma):
        """Нахождение сопряженной перестановки"""
        G = np.where(np.arange(self.n)[:, None] != np.array(sigma)[None, :],
                    self.C + self.M,
                    0)
        row_ind, col_ind = linear_sum_assignment(-G)
        conjugate_sigma = col_ind[np.argsort(row_ind)].tolist()
        S = sum(self.C[conjugate_sigma[j], j] for j in range(self.n))
        return conjugate_sigma, S

    def _find_complementary_permutation(self, sigma, gamma_vector):
        """Нахождение дополнительной перестановки"""
        G = np.where(np.arange(self.n)[:, None] != np.array(sigma)[None, :],
                    self.C + self.M,
                    np.where(np.array(gamma_vector)[None, :],
                            self.C + 2*self.M,
                            0))
        
        row_ind, col_ind = linear_sum_assignment(-G)
        sigma_comp = col_ind[np.argsort(row_ind)].tolist()
        S = sum(self.C[sigma_comp[j], j] for j in range(self.n))
        return sigma_comp, S

    def _find_best_solution(self, sigma_star, S_star, sigma_0, S0):
        """Поиск лучшего решения через перебор γ"""
        best = {'sigmas': (sigma_star, sigma_0), 'S': S_star + S0}
        
        for gamma in range(1, 2**self.n - 1):
            gamma_vector = [int(b) for b in f"{gamma:0{self.n}b}"]
            sigma_1, S1 = self._find_complementary_permutation(sigma_star, gamma_vector)
            sigma_2, S2 = self._find_conjugate_permutation(sigma_1)
            
            if S1 + S2 > best['S']:
                best = {'sigmas': (sigma_1, sigma_2), 'S': S1 + S2}
        
        return best

    def to_json(self):
        """Сериализация объекта для передачи в веб-интерфейс"""
        return json.dumps({
            'C': self.C.tolist(),
            'k': self.k,
            'n': self.n,
            'M': self.M
        })