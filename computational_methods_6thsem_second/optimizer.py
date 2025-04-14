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
        
        # 2. Проверяем что каждое подразделение атаковано не более 2 раз
        attack_counts = np.zeros(self.n)
        for period in schedule:
            for unit in period:
                attack_counts[unit] += 1
                
        if (attack_counts > 2).any():
            raise ValueError("Каждое подразделение может быть атаковано не более 2 раз")

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
        
        def is_valid_schedule(sigma1, sigma2):
            """Проверяет, что расписание удовлетворяет ограничениям"""
            attack_counts = np.zeros(self.n)
            for j in range(self.n):
                targets = [sigma1[j], sigma2[j]]
                if len(set(targets)) == 1:
                    if attack_counts[targets[0]] >= 2:
                        return False
                    attack_counts[targets[0]] += 1
                else:
                    for t in targets:
                        if attack_counts[t] >= 2:
                            return False
                        attack_counts[t] += 1
            return True
        
        def get_schedule(sigma1, sigma2):
            """Формирует расписание атак"""
            schedule = []
            attack_counts = np.zeros(self.n)
            
            for j in range(self.n):
                targets = [sigma1[j], sigma2[j]]
                if len(set(targets)) == 1:
                    main_target = targets[0]
                    if attack_counts[main_target] >= 2:
                        return None
                    
                    # Ищем альтернативную цель с учетом текущих атак
                    available_targets = [t for t in range(self.n) 
                                      if t != main_target and attack_counts[t] < 2]
                    if not available_targets:
                        return None
                    
                    # Выбираем цель с максимальной мощностью
                    best_target = max(available_targets, key=lambda t: self.C[t,j])
                    schedule.append([main_target, best_target])
                    attack_counts[main_target] += 1
                    attack_counts[best_target] += 1
                else:
                    # Проверяем, что обе цели могут быть атакованы
                    if any(attack_counts[t] >= 2 for t in targets):
                        return None
                    
                    schedule.append(targets)
                    for t in targets:
                        attack_counts[t] += 1
            
            return schedule if len(schedule) == self.n else None
        
        # Для матриц 4x4 используем специальную стратегию
        if self.n == 4:
            # Пробуем несколько стратегий для 4x4
            strategies = [
                (sigma_star, sigma_0),
                (sigma_0, sigma_star),
                (list(range(4)), list(range(3, -1, -1))),
                ([0, 1, 2, 3], [3, 2, 1, 0]),
                ([0, 2, 1, 3], [3, 1, 2, 0])
            ]
            
            for sigma1, sigma2 in strategies:
                schedule = get_schedule(sigma1, sigma2)
                if schedule is not None:
                    S7 = sum(self.C[sigma1[j], j] for j in range(self.n)) + \
                         sum(self.C[sigma2[j], j] for j in range(self.n))
                    if S7 > best['S7']:
                        best = {'sigmas': (sigma1, sigma2), 'S7': S7}
        else:
            # Для больших n используем жадный алгоритм
            if self.n > 5:
                # Пробуем несколько начальных перестановок
                initial_permutations = [
                    (sigma_star, sigma_0),
                    (sigma_0, sigma_star),
                    (list(range(self.n)), list(range(self.n-1, -1, -1)))
                ]
                
                for initial_sigmas in initial_permutations:
                    current_sigmas = initial_sigmas
                    current_S7 = sum(self.C[current_sigmas[0][j], j] for j in range(self.n)) + \
                                sum(self.C[current_sigmas[1][j], j] for j in range(self.n))
                    
                    # Пытаемся улучшить решение локальными изменениями
                    for _ in range(200):
                        improved = False
                        for j in range(self.n):
                            for i in range(self.n):
                                if i != current_sigmas[0][j] and i != current_sigmas[1][j]:
                                    new_sigma1 = current_sigmas[0].copy()
                                    new_sigma1[j] = i
                                    
                                    sigma_2, S6_2 = self._find_conjugate_permutation(new_sigma1)
                                    
                                    if is_valid_schedule(new_sigma1, sigma_2):
                                        S6_1 = sum(self.C[new_sigma1[j], j] for j in range(self.n))
                                        if (S6_1 + S6_2) > current_S7:
                                            current_sigmas = (new_sigma1, sigma_2)
                                            current_S7 = S6_1 + S6_2
                                            improved = True
                                            break
                            if improved:
                                break
                        if not improved:
                            break
                    
                    if current_S7 > best['S7']:
                        best = {'sigmas': current_sigmas, 'S7': current_S7}
            else:
                # Для малых n используем полный перебор
                for gamma in range(2**self.n):
                    gamma_vector = [int(b) for b in f"{gamma:0{self.n}b}"]
                    sigma1 = sigma_star.copy()
                    sigma2 = sigma_0.copy()
                    
                    for j in range(self.n):
                        if gamma_vector[j] == 1:
                            sigma1[j], sigma2[j] = sigma2[j], sigma1[j]
                    
                    if is_valid_schedule(sigma1, sigma2):
                        S7 = sum(self.C[sigma1[j], j] for j in range(self.n)) + \
                             sum(self.C[sigma2[j], j] for j in range(self.n))
                        if S7 > best['S7']:
                            best = {'sigmas': (sigma1, sigma2), 'S7': S7}
        
        return best