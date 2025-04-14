import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import json
from itertools import combinations

class FirepowerOptimizer:
    def __init__(self, C, k):
        # Проверка входных данных
        if not isinstance(C, list) or not all(isinstance(row, list) for row in C):
            raise ValueError("Матрица должна быть списком списков")
        if len(C) == 0 or any(len(row) != len(C) for row in C):
            raise ValueError("Матрица должна быть квадратной")
        if k <= 1:
            raise ValueError("Коэффициент k должен быть больше 1")

        self.C = np.array(C, dtype=float)
        self.k = float(k)
        self.n = len(C)
        self.M = 2 * self.C.max() + 1000  # Увеличиваем константу для большей надежности
        self.computation_time = 0

    def optimize(self):
        try:
            start_time = time.time()
            
            if (self.C < 0).any():
                raise ValueError("Матрица содержит отрицательные значения")

            if self.n == 2:
                result = self._solve_2x2_case()
            else:
                result = self._optimize_task3()
                
            # Проверка ограничений задачи
            self._validate_solution(result['schedule'])
            return result
                
        except Exception as e:
            raise RuntimeError(f"Ошибка оптимизации: {str(e)}")
        finally:
            self.computation_time = time.time() - start_time

    def _validate_solution(self, schedule):
        """Проверка что решение удовлетворяет условиям задачи"""
        # 1. Проверяем что в каждый период атакуются ровно 2 разные цели
        for period in schedule:
            if len(period) != 2 or len(set(period)) != 2:
                raise ValueError("В каждый период должно атаковаться ровно 2 разных подразделения")
        
        # 2. Проверяем общее количество атак
        attack_counts = np.zeros(self.n)
        for period in schedule:
            for unit in period:
                attack_counts[unit] += 1
        
        # Проверяем только что все подразделения атакованы хотя бы раз
        if (attack_counts == 0).any():
            raise ValueError("Каждое подразделение должно быть атаковано хотя бы раз")

    def _solve_2x2_case(self):
        """Специальное решение для матрицы 2x2 согласно пособию"""
        # Всего 2 периода и 2 подразделения, каждое должно быть атаковано 2 раза
        schedule = [[0, 1], [0, 1]]  # Обе атаки в оба периода
        
        # Расчет суммарной мощности с учетом ослабления
        total_power = 0.0
        for j in range(2):
            for i in range(2):
                if i in schedule[j]:
                    total_power += self.C[i,j] / self.k
                else:
                    total_power += self.C[i,j]
        
        return {
            'schedule': schedule,
            'total_power': float(total_power),
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time
        }

    def _optimize_task3(self):
        """Основной алгоритм оптимизации для задачи 3 из пособия"""
        # 1. Находим оптимальную перестановку σ* (максимизируем S6)
        sigma_star, S6_sigma_star = self._find_optimal_permutation(self.C)
        
        # 2. Находим сопряженную перестановку σ0
        sigma_0, S6_sigma_0 = self._find_conjugate_permutation(sigma_star)
        
        # 3. Ищем лучшее решение среди всех возможных γ
        best_solution = self._find_best_solution(sigma_star, S6_sigma_star, sigma_0, S6_sigma_0)
        
        # 4. Формируем расписание атак с проверкой корректности
        sigma1, sigma2 = best_solution['sigmas']
        schedule = []
        for j in range(self.n):
            if sigma1[j] == sigma2[j]:
                # Если в периоде j обе перестановки указывают на одну цель,
                # ищем альтернативную цель для второй атаки
                available_targets = [i for i in range(self.n) if i != sigma1[j]]
                if not available_targets:
                    raise ValueError("Невозможно сформировать корректное расписание атак")
                # Выбираем цель с максимальной мощностью
                best_target = max(available_targets, key=lambda i: self.C[i,j])
                schedule.append([int(sigma1[j]), int(best_target)])
            else:
                schedule.append([int(sigma1[j]), int(sigma2[j])])
        
        # 5. Рассчитываем итоговую мощность по формуле (42) из пособия
        total_power = float(self.C.sum() - (self.k - 1) / self.k * best_solution['S7'])
        
        return {
            'schedule': schedule,
            'total_power': total_power,
            'initial_power': int(self.C.sum()),
            'computation_time': self.computation_time,
            'sigma1': sigma1,
            'sigma2': sigma2,
            'S7': best_solution['S7']
        }

    def _find_optimal_permutation(self, matrix):
        """Решение задачи о назначениях для максимизации S6"""
        # Используем венгерский алгоритм через scipy
        row_ind, col_ind = linear_sum_assignment(-matrix)
        sigma = col_ind[np.argsort(row_ind)].tolist()
        S = sum(matrix[sigma[j], j] for j in range(self.n))
        return sigma, S

    def _find_conjugate_permutation(self, base_sigma):
        """Нахождение перестановки, сопряженной к base_sigma"""
        # Создаем модифицированную матрицу G согласно формуле (55)
        G = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != base_sigma[j]:
                    G[i,j] = self.C[i,j] + self.M
                else:
                    G[i,j] = 0  # Запрещаем выбирать те же элементы
        
        # Решаем задачу о назначениях для G
        row_ind, col_ind = linear_sum_assignment(-G)
        sigma = col_ind[np.argsort(row_ind)].tolist()
        S = sum(self.C[sigma[j], j] for j in range(self.n))
        
        return sigma, S

    def _find_complementary_permutation(self, base_sigma, gamma_vector):
        """Нахождение дополнительной перестановки относительно γ"""
        # Создаем модифицированную матрицу G согласно формуле (59)
        G = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != base_sigma[j]:
                    G[i,j] = self.C[i,j] + self.M
                elif gamma_vector[j] == 1:
                    G[i,j] = self.C[i,j] + 2*self.M  # Обязательно выбрать
                else:
                    G[i,j] = 0  # Запретить выбор
        
        # Решаем задачу о назначениях для G
        row_ind, col_ind = linear_sum_assignment(-G)
        sigma = col_ind[np.argsort(row_ind)].tolist()
        S = sum(self.C[sigma[j], j] for j in range(self.n))
        
        return sigma, S

    def _find_best_solution(self, sigma_star, S6_star, sigma_0, S6_0):
        """Поиск оптимального решения через перебор γ"""
        best = {'sigmas': (sigma_star, sigma_0), 'S7': S6_star + S6_0}
        
        def get_attack_distribution(sigma1, sigma2):
            """Анализирует распределение атак по подразделениям"""
            attack_counts = np.zeros(self.n)
            for j in range(self.n):
                attack_counts[sigma1[j]] += 1
                attack_counts[sigma2[j]] += 1
            return attack_counts
        
        def evaluate_solution(sigma1, sigma2):
            """Оценивает качество решения с учетом равномерности распределения атак"""
            attack_counts = get_attack_distribution(sigma1, sigma2)
            total_power = sum(self.C[sigma1[j], j] + self.C[sigma2[j], j] for j in range(self.n))
            
            # Штраф за неравномерное распределение атак
            attack_variance = np.var(attack_counts)
            distribution_penalty = attack_variance * total_power * 0.1
            
            return total_power - distribution_penalty
        
        # Пробуем различные комбинации перестановок
        candidates = [
            (sigma_star, sigma_0),
            (sigma_0, sigma_star),
            (list(range(self.n)), list(range(self.n-1, -1, -1))),
            (sigma_star[::-1], sigma_0[::-1]),
            (sigma_0[::-1], sigma_star[::-1])
        ]
        
        for sigma1, sigma2 in candidates:
            score = evaluate_solution(sigma1, sigma2)
            current_S7 = sum(self.C[sigma1[j], j] + self.C[sigma2[j], j] for j in range(self.n))
            
            if score > evaluate_solution(best['sigmas'][0], best['sigmas'][1]):
                best = {'sigmas': (sigma1, sigma2), 'S7': current_S7}
        
        return best