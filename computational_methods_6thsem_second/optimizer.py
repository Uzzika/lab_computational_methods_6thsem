import numpy as np
from scipy.optimize import linear_sum_assignment
import time

class FirepowerOptimizer:
    def __init__(self, C, k):
        self.C = np.array(C, dtype=int)
        self.k = k
        self.n = len(C)
        self.M = 2 * self.C.max() + 1
        self.computation_time = 0
        self.intermediate_results = []

    def optimize(self):
        if self.n > 6:
            raise ValueError("Размер матрицы слишком велик для данного алгоритма")

        try:
            start_time = time.time()
            results = self._optimize_task3()
            self.computation_time = time.time() - start_time
            return results
            
        except Exception as e:
            raise RuntimeError(f"Ошибка оптимизации: {str(e)}")

    def _optimize_task3(self):
        """Задача 3/4: эффект на 1 период, 2 выстрела за период"""
        sigma_star, S6_sigma_star = self._find_optimal_permutation(self.C)
        sigma_0, S6_sigma_0 = self._find_conjugate_permutation(sigma_star)
        best_solution = self._find_best_solution(sigma_star, S6_sigma_star, sigma_0, S6_sigma_0)
        
        self.intermediate_results.append({
            'type': 'Задача 3/4',
            'sigma*': sigma_star,
            'S6(sigma*)': S6_sigma_star,
            'sigma0': sigma_0,
            'S6(sigma0)': S6_sigma_0,
            'best_solution': best_solution
        })
        
        sigma1, sigma2 = best_solution['sigmas']
        schedule = [[sigma1[j], sigma2[j]] for j in range(self.n)]
        total_power = self.C.sum() - (self.k - 1) / self.k * best_solution['S']
        
        return {
        'schedule': schedule,
        'total_power': float(total_power),
        'initial_power': int(self.C.sum()),
        'computation_time': self.computation_time,
        'intermediate_results': self.intermediate_results
    }

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