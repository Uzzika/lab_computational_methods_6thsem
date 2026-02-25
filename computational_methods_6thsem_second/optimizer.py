import numpy as np
from scipy.optimize import linear_sum_assignment
import time

class FirepowerOptimizer:
    def __init__(self, C, k, mode='task3', strategy='optimal'):
        if not isinstance(C, list) or not all(isinstance(row, list) for row in C):
            raise ValueError("Матрица должна быть списком списков")
        if len(C) == 0 or any(len(row) != len(C) for row in C):
            raise ValueError("Матрица должна быть квадратной")
        if len(C) > 10:
            raise ValueError("Максимальный размер матрицы - 10×10")
        if k <= 1:
            raise ValueError("Коэффициент k должен быть больше 1")

        self.C = np.array(C, dtype=float)
        self.k = float(k)
        self.n = len(C)
        self.M = 2 * self.C.max() + 1000
        self.computation_time = 0
        self.mode = mode
        self.strategy = strategy  # Добавлена инициализация стратегии

    def optimize(self):
        try:
            start_time = time.time()

            if self.strategy == 'optimal':
                return self._optimize_optimal()
            elif self.strategy == 'greedy':
                return self._optimize_greedy()
            elif self.strategy == 'quasi':
                return self._optimize_quasi()
                
        finally:
            self.computation_time = time.time() - start_time

    def _optimize_task1(self):
        sigma, S6 = self._find_optimal_permutation(self.C)
        schedule = [[i] for i in sigma]
        total_power = float(self.C.sum() - (self.k - 1) / self.k * S6)
        return {
            'schedule': schedule,
            'total_power': total_power,
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time,
            'sigma1': sigma,
            'sigma2': None,
            'S7': S6,
            'strategy': 'optimize'
        }

    def _optimize_task2(self):
        C_mod = self.C.copy()
        for i in range(self.n):
            for j in range(self.n - 1):
                C_mod[i, j] += self.C[i, j + 1]
        sigma, S6 = self._find_optimal_permutation(C_mod)
        schedule = [[i] for i in sigma]
        total_power = float(self.C.sum() - (self.k - 1) / self.k * S6)
        return {
            'schedule': schedule,
            'total_power': total_power,
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time,
            'sigma1': sigma,
            'sigma2': None,
            'S7': S6,
            'strategy': 'optimize'
        }

    def _solve_2x2_case(self):
        schedule = [[0, 1], [0, 1]]
        total_power = sum(self.C[i, j] / self.k for j in range(2) for i in range(2))
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
        
        if self.n <= 5:  # Для небольших матриц делаем полный перебор
            best_solution = self._exhaustive_search(sigma_star, S6_sigma_star, sigma_0, S6_sigma_0)
        else:
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
            'S7': best_solution['S7'],
            'strategy': 'optimize'
        }

    def _find_optimal_permutation(self, matrix):
        row_ind, col_ind = linear_sum_assignment(-matrix)
        sigma = col_ind[np.argsort(row_ind)].tolist()
        S = sum(matrix[sigma[j], j] for j in range(self.n))
        return sigma, S
    
    def _find_conjugate_permutation(self, sigma_star):
        gamma_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if sigma_star[j] != i:
                    gamma_matrix[i, j] = self.C[i, j]

        row_ind, col_ind = linear_sum_assignment(-gamma_matrix)
        sigma_0 = col_ind[np.argsort(row_ind)]
        S6 = sum(self.C[sigma_0[j], j] for j in range(self.n))
        return sigma_0.tolist(), S6

    def _find_best_solution(self, sigma_star, S6_star, sigma_0, S6_0):
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
    
    def _optimize_optimal(self):
        """Оптимальная стратегия с использованием венгерского алгоритма"""
        if self.n == 2:
            return self._solve_2x2_case()  # Специальное решение для 2x2
        elif self.mode == 'task1':
            return self._optimize_task1()
        elif self.mode == 'task2':
            return self._optimize_task2()
        else:
            return self._optimize_task3()

    def _optimize_greedy(self):
        """Жадная стратегия - выбор максимальных элементов на каждом шаге"""
        if self.n == 2:
            # Для 2x2 жадный алгоритм дает оптимальный результат
            return self._solve_2x2_case()
        elif self.mode == 'task1':
            return self._greedy_task1()
        elif self.mode == 'task2':
            return self._greedy_task2()
        else:
            return self._greedy_task3()

    def _optimize_quasi(self):
        """Квазиоптимальная стратегия - комбинация жадного и оптимального подходов"""
        if self.n == 2:
            # Для 2x2 используем специальное решение
            return self._solve_2x2_case()
        elif self.n == 3:
            # Для 3x3 используем комбинацию жадного и оптимального подходов
            return self._quasi_task3()  # Используем существующий метод для задачи 3
        elif self.mode == 'task1':
            return self._quasi_task1()
        elif self.mode == 'task2':
            return self._quasi_task2()
        else:
            return self._quasi_task3()

    def _greedy_task1(self):
        """Жадная стратегия для задачи 1"""
        schedule = []
        remaining_targets = set(range(self.n))
        
        for j in range(self.n):
            # Выбираем подразделение с максимальной мощностью в текущем периоде
            max_power = -1
            best_target = None
            for i in remaining_targets:
                if self.C[i, j] > max_power:
                    max_power = self.C[i, j]
                    best_target = i
            
            if best_target is not None:
                schedule.append([best_target])
                remaining_targets.remove(best_target)
            else:
                # Если все подразделения уже атакованы, выбираем любое
                best_target = np.argmax(self.C[:, j])
                schedule.append([best_target])
        
        # Вычисляем итоговую мощность
        total_power = self._calculate_total_power(schedule)
        
        return {
            'schedule': schedule,
            'total_power': float(total_power),
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time,
            'strategy': 'greedy'
        }
    #!!!
    def _greedy_task2(self):
        #"Жадная стратегия для задачи 2 (эффект на 2 периода)"
        schedule = []
        attacked_targets = set()  # Множество атакованных подразделений

        for j in range(self.n):
            max_power = -1
            best_target = None

            # Ищем подразделение с максимальной суммарной мощностью в j и j+1 периодах
            for i in range(self.n):
                if i not in attacked_targets:
                    total_power = self.C[i, j] 
                    if j > 0:
                        total_power += self.C[i, j-1] / self.k  # Учитываем влияние предыдущего удара
                    if j+1 < self.n:
                        total_power += self.C[i, j+1]  # Учитываем следующий период
                    if total_power > max_power:
                        max_power = total_power
                        best_target = i

            # Если не нашли подразделение (все атакованы), выбираем "псевдоатаку" (не влияет на мощность)
            if best_target is None:
                best_target = 0  # Можно заменить на логику пропуска атаки

            schedule.append([best_target])
            attacked_targets.add(best_target)

        # Расчет итоговой мощности с учетом ослабления
        total_power = self._calculate_total_power(schedule)
        
        return {
            'schedule': schedule,
            'total_power': float(total_power),
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time,
            'strategy': 'greedy'
        }

    #!!!
    def _greedy_task3(self):
        """Жадная стратегия для задачи 3"""
        schedule = []
        attack_counts = {i: 0 for i in range(self.n)}
    
        for j in range(self.n):
            targets = []
            attempts = 0
            max_attempts = self.n * 2  # Ограничение на количество попыток
        
            while len(targets) < 2 and attempts < max_attempts:
                max_power = -1
                best_target = None
                for i in range(self.n):
                    if attack_counts[i] < 2 and self.C[i, j] > max_power and i not in targets:
                        max_power = self.C[i, j]
                        best_target = i
            
                if best_target is not None:
                    targets.append(best_target)
                    attack_counts[best_target] += 1
                else:
                    # Если не нашли подходящие цели, выбираем любые доступные
                    for i in range(self.n):
                        if i not in targets and attack_counts[i] < 2:
                            targets.append(i)
                            attack_counts[i] += 1
                            break
            
                attempts += 1
        
            if len(targets) < 2:
                # Если не удалось найти 2 цели, добавляем любые недостающие
                for i in range(self.n):
                    if i not in targets and len(targets) < 2:
                        targets.append(i)
        
            schedule.append(targets)
    
        # Вычисляем итоговую мощность
        total_power = self._calculate_total_power(schedule)
    
        return {
            'schedule': schedule,
            'total_power': float(total_power),
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time,
            'strategy': 'greedy'
        }

    def _quasi_task1(self):
        """Квазиоптимальная стратегия для задачи 1"""
        # Используем жадный алгоритм для первых n/2 периодов
        n_half = self.n // 2
        schedule = []
        remaining_targets = set(range(self.n))
        
        for j in range(n_half):
            max_power = -1
            best_target = None
            for i in remaining_targets:
                if self.C[i, j] > max_power:
                    max_power = self.C[i, j]
                    best_target = i
            
            if best_target is not None:
                schedule.append([best_target])
                remaining_targets.remove(best_target)
            else:
                best_target = np.argmax(self.C[:, j])
                schedule.append([best_target])
        
        # Для оставшихся периодов используем оптимальное назначение
        if remaining_targets:
            sub_matrix = self.C[list(remaining_targets), n_half:]
            _, col_ind = linear_sum_assignment(-sub_matrix)
            
            for j in range(n_half, self.n):
                best_target = list(remaining_targets)[col_ind[j - n_half]]
                schedule.append([best_target])
        
        total_power = self._calculate_total_power(schedule)
        
        return {
            'schedule': schedule,
            'total_power': float(total_power),
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time,
            'strategy': 'quasi'
        }
        
    #!!!
    def _quasi_task2(self):
        #"Квазиоптимальная стратегия для задачи 2 (эффект на 2 периода)"
        schedule = []
        remaining_targets = set(range(self.n))
        n_half = self.n // 2

        # 1. Жадный выбор для первых n_half периодов
        for j in range(n_half):
            max_power = -1
            best_target = None
            
            for i in remaining_targets:
                # Суммарная мощность с учетом влияния на текущий и следующий период
                total_power = self.C[i, j] + (self.C[i, j+1] if j+1 < self.n else 0)
                if total_power > max_power:
                    max_power = total_power
                    best_target = i
            
            if best_target is not None:
                schedule.append([best_target])
                remaining_targets.remove(best_target)
            else:
                # Если все подразделения атакованы, выбираем максимальное в текущем периоде
                best_target = np.argmax(self.C[:, j])
                schedule.append([best_target])

        # 2. Оптимальное назначение для оставшихся периодов
        if remaining_targets:
            # Создаем подматрицу для оставшихся целей и периодов
            sub_matrix = np.zeros((len(remaining_targets), self.n - n_half))
            targets_list = list(remaining_targets)
            
            for i_idx, i in enumerate(targets_list):
                for j_sub in range(n_half, self.n):
                    # Учитываем влияние на следующий период
                    power = self.C[i, j_sub] + (self.C[i, j_sub+1] if j_sub+1 < self.n else 0)
                    sub_matrix[i_idx, j_sub - n_half] = power

            # Решаем задачу о назначениях
            row_ind, col_ind = linear_sum_assignment(-sub_matrix)
            for j_sub in range(n_half, self.n):
                best_target = targets_list[row_ind[j_sub - n_half]]
                schedule.append([best_target])

        # Расчет итоговой мощности
        total_power = self._calculate_total_power(schedule)
        
        return {
            'schedule': schedule,
            'total_power': float(total_power),
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time,
            'strategy': 'quasi'
        }
       
    def _quasi_task3(self):
        #"Квазиоптимальная стратегия для задачи 3 (два удара за период)"
        schedule = []
        attack_counts = {i: 0 for i in range(self.n)}
        n_half = max(1, self.n // 2)  # Гарантируем хотя бы 1 период для жадного выбора

        # 1. Жадный выбор для первых n_half периодов
        for j in range(n_half):
            targets = []
            # Выбираем 2 цели с максимальной мощностью, которые атакованы < 2 раз
            candidates = [i for i in range(self.n) if attack_counts[i] < 2]
            candidates.sort(key=lambda i: -self.C[i, j])  # Сортируем по убыванию мощности
            
            # Берём топ-2 доступных цели
            targets = candidates[:2]
            if len(targets) < 2:
                # Если не хватает целей, добираем любыми доступными
                for i in range(self.n):
                    if i not in targets and len(targets) < 2:
                        targets.append(i)
            
            # Обновляем счётчик атак
            for i in targets:
                attack_counts[i] += 1
            schedule.append(targets)

        # 2. Оптимальное назначение для оставшихся периодов (если они есть)
        if n_half < self.n:
            # Строим подматрицу для оставшихся периодов
            remaining_periods = self.n - n_half
            sub_matrix = self.C[:, n_half:]
            
            # Решаем задачу о назначениях для двух перестановок
            try:
                # Первая перестановка (максимизация суммы)
                row_ind, col_ind = linear_sum_assignment(-sub_matrix)
                sigma1 = row_ind[np.argsort(col_ind)].tolist()
                
                # Вторая перестановка (исключаем дублирование с первой)
                masked_matrix = self.C.copy()
                for j in range(remaining_periods):
                    masked_matrix[sigma1[j], n_half + j] = -np.inf  # Исключаем дублирование

                row_ind, col_ind = linear_sum_assignment(-masked_matrix[:, n_half:])
                sigma2 = col_ind[np.argsort(row_ind)].tolist()

                for j in range(remaining_periods):
                    masked_matrix[sigma1[j], j] = -np.inf  # Исключаем уже выбранные
                
                row_ind, col_ind = linear_sum_assignment(-masked_matrix)
                sigma2 = row_ind[np.argsort(col_ind)].tolist()
                
                # Формируем расписание
                for j in range(remaining_periods):
                    targets = [sigma1[j], sigma2[j]]
                    schedule.append(targets)
            except:
                # Fallback: если оптимизация не сработала, используем жадный подход
                for j in range(n_half, self.n):
                    candidates = [i for i in range(self.n) if attack_counts[i] < 2]
                    candidates.sort(key=lambda i: -self.C[i, j])
                    targets = candidates[:2]
                    if len(targets) < 2:
                        targets.extend([i for i in range(self.n) if i not in targets][:2-len(targets)])
                    for i in targets:
                        attack_counts[i] += 1
                    schedule.append(targets)

        # Проверяем, что все подразделения атакованы не более 2 раз
        assert all(v <= 2 for v in attack_counts.values()), "Нарушено ограничение на атаки"

        # Расчёт итоговой мощности
        total_power = self._calculate_total_power(schedule)
        
        return {
            'schedule': schedule,
            'total_power': float(total_power),
            'initial_power': float(self.C.sum()),
            'computation_time': self.computation_time,
            'strategy': 'quasi',
            'sigma1': [s[0] for s in schedule[n_half:]] if n_half < self.n else None,
            'sigma2': [s[1] for s in schedule[n_half:]] if n_half < self.n else None
        }
    
    #!!!
    
    def _calculate_total_power(self, schedule):
        """Вычисляет итоговую мощность для заданного расписания"""
        total = 0
        attack_counts = np.zeros((self.n, self.n))
        
        for j, targets in enumerate(schedule):
            for i in targets:
                attack_counts[i, j] += 1
        
        for j in range(self.n):
            for i in range(self.n):
                if attack_counts[i, j] > 0:
                    total += self.C[i, j] / (self.k ** attack_counts[i, j])
                else:
                    total += self.C[i, j]
        
        return total
    
    def _exhaustive_search(self, sigma_star, S6_star, sigma_0, S6_0):
        best = {'sigmas': (sigma_star, sigma_0), 'S7': S6_star + S6_0}
        
        # Генерируем все возможные перестановки
        from itertools import permutations
        for sigma1 in permutations(range(self.n)):
            sigma1 = list(sigma1)
            S6_1 = sum(self.C[sigma1[j], j] for j in range(self.n))
            
            # Находим сопряженную перестановку
            masked_matrix = np.where(
                np.array([sigma1[j] for j in range(self.n)])[:, None] == np.arange(self.n),
                -np.inf,
                self.C
            )
            row_ind, col_ind = linear_sum_assignment(-masked_matrix)
            sigma2 = col_ind[np.argsort(row_ind)].tolist()
            S6_2 = sum(self.C[sigma2[j], j] for j in range(self.n))
            
            if S6_1 + S6_2 > best['S7']:
                best = {'sigmas': (sigma1, sigma2), 'S7': S6_1 + S6_2}
        
        return best