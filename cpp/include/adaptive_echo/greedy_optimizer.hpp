#pragma once

/**
 * Adaptive Greedy-SHADE Hybrid Optimizer
 *
 * Single-phase optimizer combining greedy hill-climbing with SHADE's adaptive
 * mechanisms for exploration. Uses a focused population around the current best
 * with success-history adaptation for mutation parameters.
 *
 * NEW ADAPTIVE FEATURES:
 * - Improvement rate monitoring with sliding window metrics
 * - Meta-learning for F/Cr parameter distributions
 * - Adaptive population behavior based on convergence detection
 * - Strategy performance tracking and dynamic selection
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
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

    // NEW: Adaptive metrics for analysis
    T final_improvement_velocity = static_cast<T>(0);
    T final_improvement_acceleration = static_cast<T>(0);
    T final_mean_f = static_cast<T>(0.5);
    T final_mean_cr = static_cast<T>(0.5);
};

/**
 * Meta-Learning State for Adaptive Optimization
 *
 * Tracks improvement trends, parameter success history, and strategy performance
 * to dynamically adapt optimizer behavior.
 */
template <typename T>
struct MetaLearningState {
    // Configuration
    static constexpr int IMPROVEMENT_WINDOW_SIZE = 30;
    static constexpr int PARAMETER_HISTORY_SIZE = 50;
    static constexpr int STRATEGY_COUNT = 3;

    // Learning rates for parameter adaptation
    T parameter_learning_rate = static_cast<T>(0.3);
    T diversity_learning_rate = static_cast<T>(0.2);

    // === IMPROVEMENT RATE MONITORING ===
    std::deque<T> improvement_history;   // Sliding window of fitness improvements
    std::deque<T> best_fitness_history;  // Best fitness at each iteration
    T prev_best_fitness = static_cast<T>(1e30);
    T improvement_velocity = static_cast<T>(0);      // Rate of change
    T improvement_acceleration = static_cast<T>(0);  // Change in velocity
    T moving_average_improvement = static_cast<T>(0);
    int iterations_since_improvement = 0;
    T convergence_score = static_cast<T>(0);  // 0=exploring, 1=converged

    // === META-LEARNING FOR PARAMETERS ===
    // Successful parameter pairs with their improvements (weighted history)
    struct ParameterSuccess {
        T f, cr;
        T improvement;
        int age;  // Iterations since recorded
    };
    std::deque<ParameterSuccess> parameter_success_history;

    // Weighted means for parameter distributions
    T adaptive_mean_f = static_cast<T>(0.5);
    T adaptive_mean_cr = static_cast<T>(0.5);

    // Success-weighted standard deviations (adapt based on convergence)
    T adaptive_f_scale = static_cast<T>(0.1);
    T adaptive_cr_std = static_cast<T>(0.1);

    // === STRATEGY PERFORMANCE TRACKING ===
    // Success counts for each mutation strategy
    std::array<int, STRATEGY_COUNT> strategy_successes = {0, 0, 0};
    std::array<int, STRATEGY_COUNT> strategy_attempts = {0, 0, 0};
    std::array<T, STRATEGY_COUNT> strategy_success_rates = {
        static_cast<T>(0.33), static_cast<T>(0.33), static_cast<T>(0.34)};

    // Strategy selection weights (for roulette wheel selection)
    std::array<T, STRATEGY_COUNT> strategy_weights = {static_cast<T>(1.0), static_cast<T>(1.0),
                                                      static_cast<T>(1.0)};

    // === ADAPTIVE POPULATION BEHAVIOR ===
    T diversity_injection_strength = static_cast<T>(1.0);  // Scales diversity injection
    T mutation_scale_boost = static_cast<T>(1.0);          // Boost mutation when stuck
    int stagnation_window_count = 0;                       // Consecutive low-improvement iterations

    // === CONVENIENCE METHODS ===

    void initialize(T initial_best_fitness) {
        prev_best_fitness = initial_best_fitness;
        improvement_history.clear();
        best_fitness_history.clear();
        parameter_success_history.clear();
        adaptive_mean_f = static_cast<T>(0.5);
        adaptive_mean_cr = static_cast<T>(0.5);
        adaptive_f_scale = static_cast<T>(0.1);
        adaptive_cr_std = static_cast<T>(0.1);
        diversity_injection_strength = static_cast<T>(1.0);
        mutation_scale_boost = static_cast<T>(1.0);
        stagnation_window_count = 0;
        iterations_since_improvement = 0;
        improvement_velocity = static_cast<T>(0);
        improvement_acceleration = static_cast<T>(0);
        convergence_score = static_cast<T>(0);

        strategy_successes = {0, 0, 0};
        strategy_attempts = {0, 0, 0};
        strategy_success_rates = {static_cast<T>(0.33), static_cast<T>(0.33), static_cast<T>(0.34)};
        strategy_weights = {static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0)};
    }

    void record_iteration(T current_best_fitness, const std::vector<T>& successful_f,
                          const std::vector<T>& successful_cr,
                          const std::vector<T>& fitness_improvements) {
        // Calculate improvement from previous best
        T improvement = prev_best_fitness - current_best_fitness;
        if (improvement < 0) improvement = static_cast<T>(0);

        // Add to sliding window
        improvement_history.push_back(improvement);
        if (improvement_history.size() > IMPROVEMENT_WINDOW_SIZE) {
            improvement_history.pop_front();
        }

        best_fitness_history.push_back(current_best_fitness);
        if (best_fitness_history.size() > IMPROVEMENT_WINDOW_SIZE) {
            best_fitness_history.pop_front();
        }

        // Update velocity and acceleration
        update_improvement_metrics();

        // Record successful parameters
        record_parameter_success(successful_f, successful_cr, fitness_improvements);

        // Update convergence detection
        if (improvement > static_cast<T>(1e-10)) {
            iterations_since_improvement = 0;
        } else {
            iterations_since_improvement++;
        }

        prev_best_fitness = current_best_fitness;
    }

    void update_improvement_metrics() {
        if (improvement_history.size() < 2) return;

        // Calculate moving average
        T sum = std::accumulate(improvement_history.begin(), improvement_history.end(),
                                static_cast<T>(0));
        moving_average_improvement = sum / static_cast<T>(improvement_history.size());

        // Calculate velocity (average rate of change over window)
        if (best_fitness_history.size() >= 2) {
            T fitness_change = best_fitness_history.front() - best_fitness_history.back();
            T window_iters = static_cast<T>(best_fitness_history.size());
            T new_velocity = fitness_change / window_iters;

            // Acceleration is change in velocity
            improvement_acceleration = new_velocity - improvement_velocity;
            improvement_velocity = new_velocity;
        }

        // Convergence score: high when velocity is low and we're near plateaus
        T velocity_norm = std::abs(improvement_velocity) /
                          (std::abs(best_fitness_history.back()) + static_cast<T>(1));
        T stagnation_factor =
            std::min(static_cast<T>(1), static_cast<T>(iterations_since_improvement) /
                                            static_cast<T>(IMPROVEMENT_WINDOW_SIZE / 2));
        convergence_score =
            std::clamp(static_cast<T>(0.5) *
                               (static_cast<T>(1) - std::tanh(velocity_norm * static_cast<T>(10))) +
                           static_cast<T>(0.5) * stagnation_factor,
                       static_cast<T>(0), static_cast<T>(1));

        // Update stagnation window count for diversity control
        if (moving_average_improvement < static_cast<T>(1e-8) ||
            iterations_since_improvement > IMPROVEMENT_WINDOW_SIZE / 3) {
            stagnation_window_count++;
        } else {
            stagnation_window_count = std::max(0, stagnation_window_count - 2);  // Decay faster
        }

        // Adapt diversity injection strength based on stagnation
        if (stagnation_window_count > 5) {
            diversity_injection_strength =
                std::min(static_cast<T>(3.0), diversity_injection_strength *
                                                  (static_cast<T>(1) + diversity_learning_rate));
            mutation_scale_boost =
                std::min(static_cast<T>(2.0),
                         mutation_scale_boost *
                             (static_cast<T>(1) + diversity_learning_rate * static_cast<T>(0.5)));
        } else if (stagnation_window_count == 0) {
            // Gradually return to normal when improving
            diversity_injection_strength =
                std::max(static_cast<T>(1.0),
                         diversity_injection_strength *
                             (static_cast<T>(1) - diversity_learning_rate * static_cast<T>(0.3)));
            mutation_scale_boost =
                std::max(static_cast<T>(1.0),
                         mutation_scale_boost *
                             (static_cast<T>(1) - diversity_learning_rate * static_cast<T>(0.3)));
        }

        // Adapt parameter distribution spread based on convergence
        if (convergence_score > static_cast<T>(0.7)) {
            // Near convergence: reduce spread for fine-tuning
            adaptive_f_scale =
                std::max(static_cast<T>(0.03), adaptive_f_scale * static_cast<T>(0.95));
            adaptive_cr_std =
                std::max(static_cast<T>(0.03), adaptive_cr_std * static_cast<T>(0.95));
        } else if (convergence_score < static_cast<T>(0.3) && stagnation_window_count > 3) {
            // Exploring or stuck: increase spread for exploration
            adaptive_f_scale =
                std::min(static_cast<T>(0.3), adaptive_f_scale * static_cast<T>(1.05));
            adaptive_cr_std =
                std::min(static_cast<T>(0.25), adaptive_cr_std * static_cast<T>(1.05));
        }
    }

    void record_parameter_success(const std::vector<T>& successful_f,
                                  const std::vector<T>& successful_cr,
                                  const std::vector<T>& fitness_improvements) {
        // Add successful parameters to history with improvement weights
        for (size_t i = 0; i < successful_f.size() && i < successful_cr.size(); ++i) {
            ParameterSuccess record;
            record.f = successful_f[i];
            record.cr = successful_cr[i];
            record.improvement =
                (i < fitness_improvements.size()) ? fitness_improvements[i] : static_cast<T>(0);
            record.age = 0;

            parameter_success_history.push_back(record);
        }

        // Age existing records and trim
        for (auto& record : parameter_success_history) {
            record.age++;
        }

        while (parameter_success_history.size() > static_cast<size_t>(PARAMETER_HISTORY_SIZE)) {
            parameter_success_history.pop_front();
        }

        // Update adaptive means based on weighted recent successes
        if (!parameter_success_history.empty()) {
            T weighted_f_sum = 0, weighted_cr_sum = 0, weight_sum = 0;

            for (const auto& record : parameter_success_history) {
                // Recent and high-improvement records get more weight
                T age_decay = std::exp(-static_cast<T>(0.1) * static_cast<T>(record.age));
                T weight = record.improvement * age_decay + static_cast<T>(1e-6);

                weighted_f_sum += weight * record.f;
                weighted_cr_sum += weight * record.cr;
                weight_sum += weight;
            }

            if (weight_sum > 0) {
                T new_mean_f = weighted_f_sum / weight_sum;
                T new_mean_cr = weighted_cr_sum / weight_sum;

                // Smooth update with learning rate
                adaptive_mean_f += parameter_learning_rate * (new_mean_f - adaptive_mean_f);
                adaptive_mean_cr += parameter_learning_rate * (new_mean_cr - adaptive_mean_cr);

                // Clamp to valid ranges
                adaptive_mean_f =
                    std::clamp(adaptive_mean_f, static_cast<T>(0.1), static_cast<T>(0.9));
                adaptive_mean_cr =
                    std::clamp(adaptive_mean_cr, static_cast<T>(0.1), static_cast<T>(0.9));
            }
        }
    }

    void record_strategy_result(int strategy_id, bool success) {
        if (strategy_id < 0 || strategy_id >= STRATEGY_COUNT) return;

        strategy_attempts[strategy_id]++;
        if (success) {
            strategy_successes[strategy_id]++;
        }

        // Update success rates with exponential moving average
        T alpha = static_cast<T>(0.3);  // Smoothing factor
        T current_rate = strategy_attempts[strategy_id] > 0
                             ? static_cast<T>(strategy_successes[strategy_id]) /
                                   static_cast<T>(strategy_attempts[strategy_id])
                             : static_cast<T>(0.5);

        strategy_success_rates[strategy_id] =
            alpha * current_rate +
            (static_cast<T>(1) - alpha) * strategy_success_rates[strategy_id];

        // Update selection weights (add small constant for exploration)
        T min_rate =
            *std::min_element(strategy_success_rates.begin(), strategy_success_rates.end());
        T max_rate =
            *std::max_element(strategy_success_rates.begin(), strategy_success_rates.end());
        T rate_range = std::max(max_rate - min_rate, static_cast<T>(0.01));

        for (int i = 0; i < STRATEGY_COUNT; ++i) {
            // Normalize and add 10% exploration bonus
            T normalized = (strategy_success_rates[i] - min_rate) / rate_range;
            strategy_weights[i] = static_cast<T>(0.1) + normalized;
        }
    }

    int select_strategy(std::mt19937& rng) const {
        std::discrete_distribution<int> dist(strategy_weights.begin(), strategy_weights.end());
        return dist(rng);
    }

    bool should_inject_diversity() const {
        // Inject diversity when stagnating and not near convergence
        return stagnation_window_count > 3 && convergence_score < static_cast<T>(0.8);
    }

    int get_diversity_count(int population_size) const {
        // Reinitialize more individuals when more stagnated
        T diversity_ratio = std::min(
            static_cast<T>(0.5),
            static_cast<T>(0.1) + static_cast<T>(stagnation_window_count) * static_cast<T>(0.05));
        return std::max(1, static_cast<int>(diversity_ratio * static_cast<T>(population_size)));
    }
};

/**
 * Adaptive Greedy-SHADE Hybrid Optimizer
 *
 * Features:
 * - Small focused population around current best (not full diversity population)
 * - SHADE success-history adaptation for F and Cr parameters
 * - DE/current-to-pbest/1 mutation for better exploration
 * - External archive for diversity maintenance
 * - Greedy selection: always move to best improvement
 * - Single phase: no separate refinement stage
 *
 * NEW: Meta-learning state for adaptive parameter control and behavior adjustment
 */
template <typename T, typename LossFn, typename SynthFn>
inline GreedyResult<T> run_greedy_optimization(
    LossFn& loss_fn, const std::vector<T>& time, SynthFn synth_fn, int num_candidates = 32,
    T initial_sigma = static_cast<T>(2.0), T time_limit = static_cast<T>(30.0),
    int shade_memory_size = 4, int archive_multiplier = 2, int stagnation_threshold = 30,
    T cr_std = static_cast<T>(0.1), T f_scale = static_cast<T>(0.1), bool verbose = true) {
    const int max_iterations = 100000;
    const int num_settings = adaptive_echo::constants::NUM_SETTINGS;
    auto t_start = std::chrono::steady_clock::now();

    auto& rng = detail::get_greedy_rng();

    // Initialize meta-learning state
    MetaLearningState<T> meta_state;

    // SHADE Memory for adaptive mutation parameters (now influenced by meta-learning)
    const int H = shade_memory_size;
    std::vector<T> M_cr(H, static_cast<T>(0.5));
    std::vector<T> M_f(H, static_cast<T>(0.5));
    int k_memory = 0;

    // Small archive for diversity
    const int archive_size = num_candidates * archive_multiplier;
    std::vector<std::vector<T>> archive;
    archive.reserve(archive_size);

    // Focused population
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

    // Initial evaluation (sequential to avoid race conditions in synth/loss functions)
    for (int i = 0; i < num_candidates; ++i) {
        fitness[i] = evaluate(population[i]);
    }

    // Find initial best
    std::vector<int> sorted_idx(num_candidates);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](int a, int b) { return fitness[a] < fitness[b]; });

    if (sorted_idx[0] != 0) {
        std::swap(population[0], population[sorted_idx[0]]);
        std::swap(fitness[0], fitness[sorted_idx[0]]);
    }

    // Initialize meta-learning state with initial best fitness
    meta_state.initialize(fitness[0]);

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
    std::vector<int> trial_strategy(num_candidates);  // Track which strategy was used

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

        // Re-sort for p-best selection
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) { return fitness[a] < fitness[b]; });

        int p_best_count = std::max(2, num_candidates / 4);

        // Blend SHADE memory with meta-learning adaptive means
        // As meta-learning confidence increases, weight toward adaptive means
        T meta_influence = std::min(static_cast<T>(0.5), static_cast<T>(it) / static_cast<T>(200));
        for (int h = 0; h < H; ++h) {
            M_f[h] = (static_cast<T>(1) - meta_influence) * M_f[h] +
                     meta_influence * meta_state.adaptive_mean_f;
            M_cr[h] = (static_cast<T>(1) - meta_influence) * M_cr[h] +
                      meta_influence * meta_state.adaptive_mean_cr;
        }

        {
            auto& thread_rng = detail::get_greedy_rng();
            std::uniform_int_distribution<int> p_best_dist(0, p_best_count - 1);
            std::uniform_int_distribution<int> pop_dist(0, num_candidates - 1);
            std::uniform_int_distribution<int> mem_dist(0, H - 1);
            std::uniform_real_distribution<T> u01(0, 1);

            for (int i = 0; i < num_candidates; ++i) {
                // Sample F and Cr using meta-learning adaptive scales
                int r_mem = mem_dist(thread_rng);
                T current_cr_std = std::max(cr_std, meta_state.adaptive_cr_std);
                T current_f_scale = std::max(f_scale, meta_state.adaptive_f_scale) *
                                    meta_state.mutation_scale_boost;

                T cr =
                    std::clamp(std::normal_distribution<T>(M_cr[r_mem], current_cr_std)(thread_rng),
                               static_cast<T>(0), static_cast<T>(1));
                T f;
                do {
                    f = std::cauchy_distribution<T>(M_f[r_mem], current_f_scale)(thread_rng);
                } while (f <= 0);
                if (f > 1) f = 1;

                trial_f[i] = f;
                trial_cr[i] = cr;

                // Select mutation strategy based on performance tracking
                int strategy = meta_state.select_strategy(thread_rng);
                trial_strategy[i] = strategy;

                // Select indices
                int pbest_idx = sorted_idx[p_best_dist(thread_rng)];
                int r1 = pop_dist(thread_rng);
                while (r1 == i) r1 = pop_dist(thread_rng);

                int combined_size = num_candidates + static_cast<int>(archive.size());
                std::uniform_int_distribution<int> combined_dist(0, std::max(0, combined_size - 1));
                int r2 = combined_dist(thread_rng);
                while (r2 == i || r2 == r1) r2 = combined_dist(thread_rng);

                const std::vector<T>& x_r2 =
                    (r2 < num_candidates) ? population[r2] : archive[r2 - num_candidates];

                // Apply selected mutation strategy
                int j_rand = std::uniform_int_distribution<int>(0, num_settings - 1)(thread_rng);

                if (strategy == 0) {
                    // DE/current-to-pbest/1: move toward pbest and away from random
                    for (int j = 0; j < num_settings; ++j) {
                        if (u01(thread_rng) < cr || j == j_rand) {
                            // Bounds check to prevent heap-buffer-overflow
                            if (j >= static_cast<int>(population[i].size()) ||
                                j >= static_cast<int>(population[pbest_idx].size()) ||
                                j >= static_cast<int>(population[r1].size()) ||
                                j >= static_cast<int>(x_r2.size()) ||
                                j >= static_cast<int>(trials[i].size()))
                                continue;
                            trials[i][j] = population[i][j] +
                                           f * (population[pbest_idx][j] - population[i][j]) +
                                           f * (population[r1][j] - x_r2[j]);
                        } else if (j < static_cast<int>(trials[i].size())) {
                            trials[i][j] = population[i][j];
                        }
                    }
                } else if (strategy == 1) {
                    // DE/best/2: more exploitation toward best
                    // Note: r2 can reference archive, so we must use x_r2 instead of population[r2]
                    int r3 = pop_dist(thread_rng);
                    while (r3 == i || r3 == r1 || r3 == r2) r3 = pop_dist(thread_rng);
                    int r4 = pop_dist(thread_rng);
                    while (r4 == i || r4 == r1 || r4 == r2 || r4 == r3) r4 = pop_dist(thread_rng);

                    // Get references for r3 and r4 (always from population)
                    const std::vector<T>& x_r3 = population[r3];
                    const std::vector<T>& x_r4 = population[r4];

                    for (int j = 0; j < num_settings; ++j) {
                        if (u01(thread_rng) < cr || j == j_rand) {
                            // Bounds check to prevent heap-buffer-overflow
                            if (j >= static_cast<int>(population[i].size()) ||
                                j >= static_cast<int>(population[0].size()) ||
                                j >= static_cast<int>(population[r1].size()) ||
                                j >= static_cast<int>(x_r2.size()) ||
                                j >= static_cast<int>(x_r3.size()) ||
                                j >= static_cast<int>(x_r4.size()) ||
                                j >= static_cast<int>(trials[i].size()))
                                continue;
                            trials[i][j] =
                                population[i][j] + f * (population[0][j] - population[i][j]) +
                                f * (population[r1][j] - x_r2[j]) + f * (x_r3[j] - x_r4[j]);
                        } else if (j < static_cast<int>(trials[i].size())) {
                            trials[i][j] = population[i][j];
                        }
                    }
                } else {
                    // DE/rand/1 with pbest influence: more exploration
                    // Note: r2 can reference archive, so we must use x_r2 instead of population[r2]
                    int r3 = pop_dist(thread_rng);
                    while (r3 == i || r3 == r1 || r3 == r2) r3 = pop_dist(thread_rng);

                    // Get reference for r3 (always from population)
                    const std::vector<T>& x_r3 = population[r3];

                    T rand_weight = static_cast<T>(0.5);
                    for (int j = 0; j < num_settings; ++j) {
                        if (u01(thread_rng) < cr || j == j_rand) {
                            // Bounds check to prevent heap-buffer-overflow
                            if (j >= static_cast<int>(population[i].size()) ||
                                j >= static_cast<int>(population[pbest_idx].size()) ||
                                j >= static_cast<int>(population[r1].size()) ||
                                j >= static_cast<int>(x_r2.size()) ||
                                j >= static_cast<int>(x_r3.size()) ||
                                j >= static_cast<int>(trials[i].size()))
                                continue;
                            trials[i][j] = population[i][j] + f * (population[r1][j] - x_r2[j]) +
                                           rand_weight * f * (population[pbest_idx][j] - x_r3[j]);
                        } else if (j < static_cast<int>(trials[i].size())) {
                            trials[i][j] = population[i][j];
                        }
                    }
                }

                trial_fitness[i] = evaluate(trials[i]);
            }
        }

        // Greedy selection and SHADE memory update
        std::vector<T> success_f, success_cr, fitness_diff;
        bool any_improvement = false;

        for (int i = 0; i < num_candidates; ++i) {
            // Record strategy performance
            meta_state.record_strategy_result(trial_strategy[i], trial_fitness[i] < fitness[i]);

            if (trial_fitness[i] < fitness[i]) {
                // Archive the old solution
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

                // Update global best
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

        // Update meta-learning state
        meta_state.record_iteration(result.best_loss, success_f, success_cr, fitness_diff);

        // ADAPTIVE POPULATION BEHAVIOR
        // Inject diversity based on meta-learning state
        if (meta_state.should_inject_diversity()) {
            int diversity_count = meta_state.get_diversity_count(num_candidates);

            // Reinitialize worst individuals with diversity
            std::normal_distribution<T> restart_dist(
                0, initial_sigma * meta_state.diversity_injection_strength);

            for (int i = 0; i < diversity_count && i < num_candidates - 1; ++i) {
                int idx = sorted_idx[num_candidates - 1 - i];  // Worst individuals
                if (idx == 0) continue;                        // Never reinitialize best

                for (int j = 0; j < num_settings; ++j) {
                    T val = std::clamp(result.best_settings[j], static_cast<T>(1e-6),
                                       static_cast<T>(1.0 - 1e-6));
                    T logit = std::log(val / (static_cast<T>(1.0) - val));
                    population[idx][j] = logit + restart_dist(rng);
                }
                fitness[idx] = evaluate(population[idx]);
            }

            if (verbose && it % 10 == 0) {
                std::cout << "  [Diversity Injection] Reinitialized " << diversity_count
                          << " individuals (strength=" << meta_state.diversity_injection_strength
                          << ")" << std::endl;
            }
        }

        // Traditional soft restart as fallback
        if (stagnation_count >= max_stagnation) {
            std::normal_distribution<T> restart_dist(0, initial_sigma);
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

            // Reset some meta-learning state on full restart
            meta_state.stagnation_window_count = 0;
            meta_state.diversity_injection_strength = static_cast<T>(1.0);
        }

        if (verbose && it % 10 == 0) {
            std::cout << "Iter " << it << ": Best Loss = " << result.best_loss
                      << " | Conv=" << std::fixed << std::setprecision(2)
                      << meta_state.convergence_score << " | F=" << meta_state.adaptive_mean_f
                      << " | Cr=" << meta_state.adaptive_mean_cr
                      << " | Vel=" << meta_state.improvement_velocity << " | Elapsed: " << elapsed
                      << "s" << std::endl;
        }
        result.iterations_completed = it + 1;
    }

    if (result.iterations_completed == 0) {
        result.iterations_completed = max_iterations;
    }

    // Populate result with final meta-learning metrics
    result.final_improvement_velocity = meta_state.improvement_velocity;
    result.final_improvement_acceleration = meta_state.improvement_acceleration;
    result.final_mean_f = meta_state.adaptive_mean_f;
    result.final_mean_cr = meta_state.adaptive_mean_cr;

    return result;
}

}  // namespace adaptive_echo
