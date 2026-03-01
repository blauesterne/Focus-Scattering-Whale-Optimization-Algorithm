"""************************************************************************************
                            The source code of FS-WOA
                It is programmed by Sungwook Cho and Seong S. Cheon
         Department of mechanical engineering, Kongju National University
                                   South Korea
                              Date: March 1st, 2026
                               All rights reserved.
                            csw5046@smail.kongju.ac.kr
*************************************************************************************"""

import  time
import numpy as np


class OptimizationAlgorithms:
    def __init__(self, opt_func, constraints, nsols, ngens):
        self.opt_func = opt_func
        self.constraints = constraints
        self.nsols = nsols
        self.ngens = ngens

    def FS_WOA(self):
        maximize = False

        lb = np.array([c[0] for c in self.constraints])
        ub = np.array([c[1] for c in self.constraints])

        def sample_zeta(mean=0.25, std=0.1, min_val=1e-6, max_val=0.5):
            val = np.random.normal(mean, std)
            return np.clip(val, min_val, max_val)

        def modified_encircle(sol, best_sol, A, c):
            C = c * np.random.rand(len(self.constraints))
            D = np.abs(C * best_sol - sol)
            return best_sol - A * D

        def exploration(sol, rand_sol, A, c):
            C = c * np.random.rand(len(self.constraints))
            D = np.abs(C * rand_sol - sol)
            return rand_sol - A * D

        def spiral(sol, best_sol):
            b = sample_zeta(mean=0.5, std=0.15, min_val=0.1, max_val=1.0)
            D = np.linalg.norm(best_sol - sol)
            L = np.random.uniform(-1.0, 1.0, size=len(self.constraints))
            return D * np.exp(b * L) * np.cos(2 * np.pi * L) + best_sol


        def Focusing_group_optimize(nsols, solutions, best_solutions, a):
            temp_new_sols = []
            for i in range(1, nsols):
                sol = solutions[i]
                p = np.random.rand()
                A = 2 * a * np.random.rand(len(self.constraints)) - a
                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        new_sol_attack = modified_encircle(sol, best_solutions, A, c=1.0)
                    else:
                        rand_idx = np.random.randint(0, nsols)
                        rand_sol = solutions[rand_idx]
                        new_sol_attack = exploration(sol, rand_sol, A, c=1.0)
                else:
                    new_sol_attack = spiral(sol, best_solutions)
                temp_new_sols.append(np.clip(new_sol_attack, lb, ub))
            return temp_new_sols


        def Scattering_group_optimize(solutions, best_solutions, a, threshold):

            sols = np.asarray(solutions, dtype=float)
            n, d = sols.shape

            if n == 0:
                return []

            span = (ub - lb)

            thr = max(float(threshold), 1e-6 * (np.linalg.norm(span) + 1e-30) / max(1, d ** 0.5))

            F = np.clip(0.55 + 0.25 * np.random.randn(n, 1), 0.3, 0.9)
            CR = np.clip(0.85 - 0.45 * (1.0 - a) + 0.05 * np.random.randn(n, 1), 0.2, 0.95)

            new = []
            for i in range(n):
                x = sols[i]

                if n >= 3:
                    pool = np.delete(np.arange(n), i)
                    r1, r2 = np.random.choice(pool, 2, replace=False)
                elif n == 2:
                    r1, r2 = 1 - i, i
                else:
                    r1 = r2 = i


                v = x + F[i] * (best_solutions - x) + F[i] * (sols[r1] - sols[r2])


                jrand = np.random.randint(0, d)
                cross_mask = (np.random.rand(d) < CR[i].item())
                cross_mask[jrand] = True
                u = np.where(cross_mask, v, x)


                if np.random.rand() < (0.15 + 0.25 * (1.0 - a)):

                    u = 0.5 * (lb + ub) + np.random.normal(0, 0.005, size=d) * span


                if np.linalg.norm(u - best_solutions) < thr:
                    u += np.random.uniform(-0.02, 0.02, size=d) * span


                u = np.clip(u, lb, ub)
                if len(new) > 0:
                    arr = np.asarray(new)
                    if np.any(np.linalg.norm(arr - u, axis=1) < thr):
                        u += np.random.normal(0, 0.01, size=d) * span
                        u = np.clip(u, lb, ub)
                new.append(u)

            return new


        sols = np.random.uniform(lb, ub, size=(self.nsols, len(self.constraints)))
        best_values = []
        obs = []

        start_time = time.perf_counter()
        for gen in range(self.ngens):
            zeta = sample_zeta()
            if len(obs) < 2:
                a_val = zeta
            else:
                obs_gradient = obs[-1] / (obs[0] + 1e-30) + zeta
                a_val = obs_gradient if obs_gradient < 0.1 else zeta
            a_aggressive = a_val * (2 - 2 * gen / self.ngens)


            fitnesses = np.array([self.opt_func(sol) for sol in sols])
            best_idx = np.argmax(fitnesses) if maximize else np.argmin(fitnesses)
            best_sol = sols[best_idx].copy()
            best_fitness = fitnesses[best_idx]
            best_values.append(best_fitness)
            obs.append(best_fitness)

            sorted_indices = np.argsort(fitnesses)
            sorted_bestsols_older = sols[sorted_indices]


            split_para = np.clip(np.random.normal(0.5, 0.5), 0.0, 1.0)
            attack_group = int(self.nsols * split_para)

            new_sols = [best_sol]
            new_sols_attack = sorted_bestsols_older[:attack_group]
            new_sols_search = sorted_bestsols_older[attack_group:]

            new_sols.extend(Focusing_group_optimize(nsols=attack_group,
                                                  solutions=new_sols_attack,
                                                  best_solutions=best_sol,
                                                  a=a_aggressive)
                            )
            new_sols.extend(Scattering_group_optimize(solutions=new_sols_search,
                                                  best_solutions=best_sol,
                                                  a=a_aggressive,
                                                  threshold=1e-5)
                            )
            sols = np.stack(new_sols)

        elapsed = time.perf_counter() - start_time


        return best_values, elapsed

