#pragma once

/**
 * Hybrid Evolution optimizer combining SHADE (Success-History Adaptation Differential Evolution)
 * and ES (Evolution Strategies) mechanisms for high-dimensional audio parameter optimization.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "adaptive_echo/constants.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace adaptive_echo {

namespace detail {
inline std::mt19937& get_hybrid_rng() {
    static thread_local std::mt19937 rng(std::random_device{}() + 42);
    return rng;
}
}  // namespace detail

template <typename T>
inline T sigmoid_hybrid(T x) {
    x = std::clamp(x, static_cast<T>(-500), static_cast<T>(500));
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
}

template <typename T>
struct HybridResult {
    std::vector<T> best_settings;
    T best_loss;
};

/**
 * SHADE-ES Hybrid Optimizer
 *
 * Features:
 * - Success-History Adaptation (SHADE) for F and Cr parameters
 * - DE/current-to-pbest/1 mutation strategy
 * - External Archive for diversity maintenance
 * - Persistent population (spatial memory)
 * - Thread-safe parallel evaluation
 */
template <typename T, typename LossFn, typename SynthFn>
inline HybridResult<T> run_hybrid_evolution(
    LossFn& loss_fn, const std::vector<T>& time, SynthFn synth_fn, int population_size = 128,
    int num_iterations = 10000, T sigma_init = static_cast<T>(0.8), T F_init = static_cast<T>(0.5),
    T Cr_init = static_cast<T>(0.5), T time_limit = static_cast<T>(30.0)) {
    const int num_settings = adaptive_echo::constants::NUM_SETTINGS;
    auto t_start_all = std::chrono::steady_clock::now();

    // SHADE Memory
    const int H = 6;
    std::vector<T> M_cr(H, Cr_init);
    std::vector<T> M_f(H, F_init);
    int k_memory = 0;

    // Archive
    const int archive_size = population_size;
    std::vector<std::vector<T>> archive;
    archive.reserve(archive_size);

    // Population
    std::vector<std::vector<T>> population(population_size, std::vector<T>(num_settings));
    std::vector<T> fitness(population_size);

    // Initialize population randomly in logit space
    {
        auto& rng = detail::get_hybrid_rng();
        std::uniform_real_distribution<T> init_dist(-sigma_init * 5.0, sigma_init * 5.0);
        for (int i = 0; i < population_size; ++i) {
            for (int j = 0; j < num_settings; ++j) {
                population[i][j] = init_dist(rng);
            }
        }
    }

    auto evaluate = [&](const std::vector<T>& individual) {
        std::vector<T> settings(num_settings);
        for (int j = 0; j < num_settings; ++j) {
            settings[j] = sigmoid_hybrid(individual[j]);
        }
        auto audio = synth_fn(settings, time);
        return loss_fn(audio);
    };

    // Initial evaluation
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < population_size; ++i) {
        fitness[i] = evaluate(population[i]);
    }

    HybridResult<T> result;
    result.best_loss = std::numeric_limits<T>::max();
    result.best_settings.assign(num_settings, static_cast<T>(0));

    auto update_best = [&]() {
        bool improved = false;
        for (int i = 0; i < population_size; ++i) {
            if (fitness[i] < result.best_loss) {
                result.best_loss = fitness[i];
                for (int j = 0; j < num_settings; ++j) {
                    result.best_settings[j] = sigmoid_hybrid(population[i][j]);
                }
                improved = true;
            }
        }
        return improved;
    };
    update_best();

    std::vector<std::vector<T>> trials(population_size, std::vector<T>(num_settings));
    std::vector<T> trial_fitness(population_size);
    std::vector<T> trial_f(population_size);
    std::vector<T> trial_cr(population_size);

    int stagnation = 0;

    for (int it = 0; it < num_iterations; ++it) {
        auto t_now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<T>>(t_now - t_start_all).count();
        if (time_limit > 0 && elapsed > time_limit) {
            std::cout << "Time limit reached. Stopping." << std::endl;
            break;
        }

        // Sort indices for p-best selection
        std::vector<int> sorted_idx(population_size);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) { return fitness[a] < fitness[b]; });

        int p_best_count = std::max(2, static_cast<int>(0.15 * population_size));

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
            auto& thread_rng = detail::get_hybrid_rng();
            std::uniform_int_distribution<int> p_best_dist(0, p_best_count - 1);
            std::uniform_int_distribution<int> pop_dist(0, population_size - 1);
            std::uniform_int_distribution<int> mem_dist(0, H - 1);
            std::uniform_real_distribution<T> u01(0, 1);

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
            for (int i = 0; i < population_size; ++i) {
                int r_mem = mem_dist(thread_rng);

                T cr = std::clamp(std::normal_distribution<T>(M_cr[r_mem], 0.1)(thread_rng),
                                  static_cast<T>(0), static_cast<T>(1));
                T f;
                do {
                    f = std::cauchy_distribution<T>(M_f[r_mem], 0.1)(thread_rng);
                } while (f <= 0);
                if (f > 1) f = 1;

                trial_f[i] = f;
                trial_cr[i] = cr;

                int pbest_idx = sorted_idx[p_best_dist(thread_rng)];
                int r1 = pop_dist(thread_rng);
                while (r1 == i) r1 = pop_dist(thread_rng);

                int combined_size = population_size + static_cast<int>(archive.size());
                std::uniform_int_distribution<int> combined_dist(0, combined_size - 1);
                int r2 = combined_dist(thread_rng);
                while (r2 == i || r2 == r1) r2 = combined_dist(thread_rng);

                const std::vector<T>& x_r2 =
                    (r2 < population_size) ? population[r2] : archive[r2 - population_size];

                int j_rand = std::uniform_int_distribution<int>(0, num_settings - 1)(thread_rng);
                for (int j = 0; j < num_settings; ++j) {
                    if (u01(thread_rng) < cr || j == j_rand) {
                        trials[i][j] = population[i][j] +
                                       f * (population[pbest_idx][j] - population[i][j]) +
                                       f * (population[r1][j] - x_r2[j]);
                    } else {
                        trials[i][j] = population[i][j];
                    }
                }
                trial_fitness[i] = evaluate(trials[i]);
            }
        }

        // Selection and Memory Update
        std::vector<T> success_f, success_cr, fitness_diff;
        for (int i = 0; i < population_size; ++i) {
            if (trial_fitness[i] < fitness[i]) {
                // Add to archive
                if (archive.size() < static_cast<size_t>(archive_size)) {
                    archive.push_back(population[i]);
                } else {
                    std::uniform_int_distribution<int> arch_dist(0, archive_size - 1);
                    archive[arch_dist(detail::get_hybrid_rng())] = population[i];
                }

                success_f.push_back(trial_f[i]);
                success_cr.push_back(trial_cr[i]);
                fitness_diff.push_back(fitness[i] - trial_fitness[i]);

                population[i] = trials[i];
                fitness[i] = trial_fitness[i];
            } else if (trial_fitness[i] == fitness[i]) {
                population[i] = trials[i];
            }
        }

        // Update SHADE memory
        if (!success_f.empty()) {
            T sum_diff =
                std::accumulate(fitness_diff.begin(), fitness_diff.end(), static_cast<T>(0));
            T next_m_cr = 0;
            T next_m_f_num = 0;
            T next_m_f_den = 0;

            for (size_t s = 0; s < success_f.size(); ++s) {
                T weight = fitness_diff[s] / sum_diff;
                next_m_cr += weight * success_cr[s];
                next_m_f_num += weight * success_f[s] * success_f[s];
                next_m_f_den += weight * success_f[s];
            }

            M_cr[k_memory] = next_m_cr;
            M_f[k_memory] = next_m_f_num / next_m_f_den;

            k_memory = (k_memory + 1) % H;
        }

        if (update_best()) {
            stagnation = 0;
        } else {
            stagnation++;
        }

        // Soft Restart / Diversity Injection
        if (stagnation > 20) {
            auto& rng = detail::get_hybrid_rng();
            std::normal_distribution<T> restart_dist(0, 1.5);
            // Keep only top 5%, randomize the rest around best
            for (int i = std::max(1, population_size / 20); i < population_size; ++i) {
                int idx = sorted_idx[i];
                for (int j = 0; j < num_settings; ++j) {
                    T val = std::clamp(result.best_settings[j], static_cast<T>(1e-6),
                                       static_cast<T>(1.0 - 1e-6));
                    population[idx][j] =
                        std::log(val / (static_cast<T>(1.0) - val)) + restart_dist(rng);
                }
                fitness[idx] = evaluate(population[idx]);
            }
            stagnation = 0;
            update_best();
        }

        if (it % 10 == 0) {
            std::cout << "Iter " << it << ": Best Loss = " << result.best_loss
                      << " | Elapsed: " << elapsed << "s" << std::endl;
        }
    }

    // Final Greedy Refinement (Local Search)
    auto& rng = detail::get_hybrid_rng();
    std::vector<T> current_best_logit(num_settings);
    for (int j = 0; j < num_settings; ++j) {
        current_best_logit[j] = std::log(
            std::clamp(result.best_settings[j], static_cast<T>(1e-6), static_cast<T>(1.0 - 1e-6)) /
            (1.0 - std::clamp(result.best_settings[j], static_cast<T>(1e-6),
                              static_cast<T>(1.0 - 1e-6))));
    }

    T current_sigma = 0.1;
    for (int step = 0; step < 500; ++step) {
        std::vector<T> candidate = current_best_logit;
        std::normal_distribution<T> dist(0, current_sigma);
        for (int j = 0; j < num_settings; ++j) {
            candidate[j] += dist(rng);
        }
        T candidate_loss = evaluate(candidate);
        if (candidate_loss < result.best_loss) {
            result.best_loss = candidate_loss;
            current_best_logit = candidate;
            for (int j = 0; j < num_settings; ++j) {
                result.best_settings[j] = sigmoid_hybrid(candidate[j]);
            }
            current_sigma *= 1.1;  // Success, try larger steps
        } else {
            current_sigma *= 0.5;  // Failure, shrink search
        }
        if (current_sigma < 1e-6) break;
    }

    return result;
}

}  // namespace adaptive_echo
