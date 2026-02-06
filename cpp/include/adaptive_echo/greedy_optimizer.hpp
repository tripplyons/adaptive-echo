#pragma once

/**
 * Greedy-SHADE Hybrid Optimizer
 *
 * Single-phase optimizer combining greedy hill-climbing with SHADE's adaptive
 * mechanisms for exploration. Uses a focused population around the current best
 * with success-history adaptation for mutation parameters.
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
inline std::mt19937& get_greedy_rng() {
    static thread_local std::mt19937 rng(std::random_device{}() + 42);
    return rng;
}
}  // namespace detail

template <typename T>
inline T sigmoid_greedy(T x) {
    x = std::clamp(x, static_cast<T>(-500), static_cast<T>(500));
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
}

template <typename T>
struct GreedyResult {
    std::vector<T> best_settings;
    T best_loss;
    int iterations_completed = 0;
};

/**
 * Greedy-SHADE Hybrid Optimizer
 *
 * Features:
 * - Small focused population around current best (not full diversity population)
 * - SHADE success-history adaptation for F and Cr parameters
 * - DE/current-to-pbest/1 mutation for better exploration
 * - External archive for diversity maintenance
 * - Greedy selection: always move to best improvement
 * - Single phase: no separate refinement stage
 */
template <typename T, typename LossFn, typename SynthFn>
inline GreedyResult<T> run_greedy_optimization(
    LossFn& loss_fn, const std::vector<T>& time, SynthFn synth_fn, int num_candidates = 32,
    T initial_sigma = static_cast<T>(2.0), T time_limit = static_cast<T>(30.0),
    int shade_memory_size = 4, int archive_multiplier = 2, int stagnation_threshold = 30,
    T cr_std = static_cast<T>(0.1), T f_scale = static_cast<T>(0.1), bool verbose = true) {
    const int max_iterations = 100000;  // Run until time limit reached
    const int num_settings = adaptive_echo::constants::NUM_SETTINGS;
    auto t_start = std::chrono::steady_clock::now();

    auto& rng = detail::get_greedy_rng();

    // SHADE Memory for adaptive mutation parameters
    const int H = shade_memory_size;
    std::vector<T> M_cr(H, static_cast<T>(0.5));
    std::vector<T> M_f(H, static_cast<T>(0.5));
    int k_memory = 0;

    // Small archive for diversity (stores discarded solutions)
    const int archive_size = num_candidates * archive_multiplier;
    std::vector<std::vector<T>> archive;
    archive.reserve(archive_size);

    // Focused population: current best + candidates around it
    // Population[0] is always the current best
    std::vector<std::vector<T>> population(num_candidates, std::vector<T>(num_settings));
    std::vector<T> fitness(num_candidates);

    // Initialize population randomly
    std::uniform_real_distribution<T> init_dist(-5.0, 5.0);
    for (int i = 0; i < num_candidates; ++i) {
        for (int j = 0; j < num_settings; ++j) {
            population[i][j] = init_dist(rng);
        }
    }

    auto evaluate = [&](const std::vector<T>& individual) {
        std::vector<T> settings(num_settings);
        for (int j = 0; j < num_settings; ++j) {
            settings[j] = sigmoid_greedy(individual[j]);
        }
        auto audio = synth_fn(settings, time);
        return loss_fn(audio);
    };

    // Initial evaluation
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < num_candidates; ++i) {
        fitness[i] = evaluate(population[i]);
    }

    // Find initial best and sort
    std::vector<int> sorted_idx(num_candidates);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](int a, int b) { return fitness[a] < fitness[b]; });

    // Move best to position 0
    if (sorted_idx[0] != 0) {
        std::swap(population[0], population[sorted_idx[0]]);
        std::swap(fitness[0], fitness[sorted_idx[0]]);
    }

    GreedyResult<T> result;
    result.best_loss = fitness[0];
    result.best_settings.resize(num_settings);
    for (int j = 0; j < num_settings; ++j) {
        result.best_settings[j] = sigmoid_greedy(population[0][j]);
    }

    // Working buffers
    std::vector<std::vector<T>> trials(num_candidates, std::vector<T>(num_settings));
    std::vector<T> trial_fitness(num_candidates);
    std::vector<T> trial_f(num_candidates);
    std::vector<T> trial_cr(num_candidates);

    int stagnation_count = 0;
    const int max_stagnation = stagnation_threshold;

    for (int it = 0; it < max_iterations; ++it) {
        auto t_now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<T>>(t_now - t_start).count();
        if (time_limit > 0 && elapsed > time_limit) {
            if (verbose) {
                std::cout << "Time limit reached. Stopping." << std::endl;
            }
            break;
        }

        // Re-sort for p-best selection (greedy: always use current best as reference)
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) { return fitness[a] < fitness[b]; });

        int p_best_count = std::max(2, num_candidates / 4);

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
            auto& thread_rng = detail::get_greedy_rng();
            std::uniform_int_distribution<int> p_best_dist(0, p_best_count - 1);
            std::uniform_int_distribution<int> pop_dist(0, num_candidates - 1);
            std::uniform_int_distribution<int> mem_dist(0, H - 1);
            std::uniform_real_distribution<T> u01(0, 1);

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
            for (int i = 0; i < num_candidates; ++i) {
                // Sample F and Cr from SHADE memory
                int r_mem = mem_dist(thread_rng);
                T cr = std::clamp(std::normal_distribution<T>(M_cr[r_mem], cr_std)(thread_rng),
                                  static_cast<T>(0), static_cast<T>(1));
                T f;
                do {
                    f = std::cauchy_distribution<T>(M_f[r_mem], f_scale)(thread_rng);
                } while (f <= 0);
                if (f > 1) f = 1;

                trial_f[i] = f;
                trial_cr[i] = cr;

                // DE/current-to-pbest/1 mutation
                // Population[0] is current best (greedy focus)
                int pbest_idx = sorted_idx[p_best_dist(thread_rng)];
                int r1 = pop_dist(thread_rng);
                while (r1 == i) r1 = pop_dist(thread_rng);

                // Archive diversity: sample from population + archive
                int combined_size = num_candidates + static_cast<int>(archive.size());
                std::uniform_int_distribution<int> combined_dist(0, std::max(0, combined_size - 1));
                int r2 = combined_dist(thread_rng);
                while (r2 == i || r2 == r1) r2 = combined_dist(thread_rng);

                const std::vector<T>& x_r2 =
                    (r2 < num_candidates) ? population[r2] : archive[r2 - num_candidates];

                // Binomial crossover with DE mutation
                int j_rand = std::uniform_int_distribution<int>(0, num_settings - 1)(thread_rng);
                for (int j = 0; j < num_settings; ++j) {
                    if (u01(thread_rng) < cr || j == j_rand) {
                        // DE/current-to-pbest/1: move toward pbest and away from random
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

        // Greedy selection and SHADE memory update
        std::vector<T> success_f, success_cr, fitness_diff;
        bool any_improvement = false;

        for (int i = 0; i < num_candidates; ++i) {
            if (trial_fitness[i] < fitness[i]) {
                // Archive the old solution before replacing
                if (archive.size() < static_cast<size_t>(archive_size)) {
                    archive.push_back(population[i]);
                } else if (!archive.empty()) {
                    std::uniform_int_distribution<int> arch_dist(0, archive_size - 1);
                    archive[arch_dist(rng)] = population[i];
                }

                // Track success for SHADE memory update
                success_f.push_back(trial_f[i]);
                success_cr.push_back(trial_cr[i]);
                fitness_diff.push_back(fitness[i] - trial_fitness[i]);

                population[i] = trials[i];
                fitness[i] = trial_fitness[i];
                any_improvement = true;

                // Update global best if this is best so far
                if (fitness[i] < result.best_loss) {
                    result.best_loss = fitness[i];
                    for (int j = 0; j < num_settings; ++j) {
                        result.best_settings[j] = sigmoid_greedy(population[i][j]);
                    }
                    stagnation_count = 0;
                }
            }
        }

        // Update SHADE memory with weighted Lehmer mean
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

        if (!any_improvement) {
            stagnation_count++;
        }

        // Soft restart: inject diversity when stuck
        if (stagnation_count >= max_stagnation) {
            std::normal_distribution<T> restart_dist(0, initial_sigma);
            // Keep best, randomize others around it
            for (int i = 1; i < num_candidates; ++i) {
                for (int j = 0; j < num_settings; ++j) {
                    T val = std::clamp(result.best_settings[j], static_cast<T>(1e-6),
                                       static_cast<T>(1.0 - 1e-6));
                    T logit = std::log(val / (static_cast<T>(1.0) - val));
                    population[i][j] = logit + restart_dist(rng);
                }
                fitness[i] = evaluate(population[i]);
            }
            stagnation_count = 0;
        }

        if (verbose && it % 10 == 0) {
            std::cout << "Iter " << it << ": Best Loss = " << result.best_loss
                      << " | Elapsed: " << elapsed << "s" << std::endl;
        }
        result.iterations_completed = it + 1;
    }

    if (result.iterations_completed == 0) {
        result.iterations_completed = max_iterations;
    }

    return result;
}

}  // namespace adaptive_echo
