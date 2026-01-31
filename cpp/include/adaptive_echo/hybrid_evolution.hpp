#pragma once

/**
 * Hybrid Evolution optimizer combining ES, DE, and GA mechanisms.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "adaptive_echo/constants.hpp"

namespace adaptive_echo {

namespace detail {
inline std::mt19937& get_hybrid_rng() {
    static std::mt19937 rng(42);
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

template <typename T, typename LossFn, typename SynthFn>
inline HybridResult<T> run_hybrid_evolution(
    LossFn& loss_fn, const std::vector<T>& time, SynthFn synth_fn, int population_size = 64,
    int num_iterations = 100, T sigma_init = static_cast<T>(0.6),
    T sigma_min = static_cast<T>(0.05), T sigma_max = static_cast<T>(2.0),
    T F_scale_start = static_cast<T>(0.75), T F_scale_end = static_cast<T>(0.15),
    T crossover_rate_start = static_cast<T>(0.75), T crossover_rate_end = static_cast<T>(0.35),
    T mutation_rate = static_cast<T>(0.03), T mutation_sigma = static_cast<T>(0.06),
    T elite_fraction = static_cast<T>(0.1), T time_limit = static_cast<T>(-1)) {
    const int num_settings = adaptive_echo::constants::NUM_SETTINGS;

    auto t_start_all = std::chrono::steady_clock::now();
    auto& rng = detail::get_hybrid_rng();
    std::normal_distribution<T> normal_dist(0, 1);
    std::uniform_real_distribution<T> uniform_dist(0, 1);
    std::uniform_int_distribution<int> idx_dist(0, population_size - 1);

    // ES-style mean and diagonal stddev in logit space
    std::vector<T> mean(num_settings, static_cast<T>(0));
    std::vector<T> sigma(num_settings, sigma_init);

    int mu = std::max(2, population_size / 2);
    std::vector<T> weights(mu);
    for (int i = 0; i < mu; ++i) {
        weights[i] = std::log(static_cast<T>(mu + 0.5)) - std::log(static_cast<T>(i + 1));
    }
    T w_sum = std::accumulate(weights.begin(), weights.end(), static_cast<T>(0));
    for (auto& w : weights) {
        w /= w_sum;
    }

    std::vector<std::vector<T>> population(population_size, std::vector<T>(num_settings));
    std::vector<std::vector<T>> trials(population_size, std::vector<T>(num_settings));
    std::vector<T> fitness(population_size);
    std::vector<T> trial_fitness(population_size);
    std::vector<int> indices(population_size);

    HybridResult<T> result;
    result.best_loss = std::numeric_limits<T>::max();
    result.best_settings.assign(num_settings, static_cast<T>(0));
    int stagnation = 0;

    auto evaluate = [&](const std::vector<T>& individual) {
        std::vector<T> settings(num_settings);
        for (int j = 0; j < num_settings; ++j) {
            settings[j] = sigmoid_hybrid(individual[j]);
        }
        auto audio = synth_fn(settings, time);
        return loss_fn(audio);
    };

    for (int it = 0; it < num_iterations; ++it) {
        auto t_it_start = std::chrono::steady_clock::now();

        if (time_limit > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - t_start_all)
                               .count();
            if (elapsed > time_limit) {
                std::cout << "Time limit reached (" << time_limit << "s). Stopping at iteration "
                          << it << "." << std::endl;
                break;
            }
        }

        // Sample ES population around mean/sigma
        for (int i = 0; i < population_size; ++i) {
            for (int j = 0; j < num_settings; ++j) {
                population[i][j] = mean[j] + sigma[j] * normal_dist(rng);
            }
            fitness[i] = evaluate(population[i]);
        }

        // Rank population
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return fitness[a] < fitness[b]; });

        // Track best
        if (fitness[indices[0]] < result.best_loss) {
            result.best_loss = fitness[indices[0]];
            for (int j = 0; j < num_settings; ++j) {
                result.best_settings[j] = sigmoid_hybrid(population[indices[0]][j]);
            }
            stagnation = 0;
        } else {
            stagnation += 1;
        }

        // DE-style trials using best and random differences
        T progress = static_cast<T>(it) / static_cast<T>(num_iterations);
        T current_F = F_scale_start - (F_scale_start - F_scale_end) * progress;
        T current_Cr =
            crossover_rate_start - (crossover_rate_start - crossover_rate_end) * progress;
        const auto& best_individual = population[indices[0]];
        // Adaptive mutation: decay over time, boost on stagnation
        T adaptive_mut_rate =
            mutation_rate * (static_cast<T>(1.0) - static_cast<T>(0.6) * progress);
        T adaptive_mut_sigma =
            mutation_sigma * (static_cast<T>(1.0) - static_cast<T>(0.4) * progress);
        if (stagnation >= 10) {
            adaptive_mut_rate =
                std::min(static_cast<T>(0.15), adaptive_mut_rate * static_cast<T>(2.0));
            adaptive_mut_sigma =
                std::min(static_cast<T>(0.5), adaptive_mut_sigma * static_cast<T>(2.0));
        }

        for (int i = 0; i < population_size; ++i) {
            int r1 = idx_dist(rng);
            int r2 = idx_dist(rng);
            int r0 = idx_dist(rng);
            bool use_best = uniform_dist(rng) < static_cast<T>(0.5);

            for (int j = 0; j < num_settings; ++j) {
                T base = use_best ? best_individual[j] : population[r0][j];
                T donor = base + current_F * (population[r1][j] - population[r2][j]);
                donor += normal_dist(rng) * (static_cast<T>(0.15) * current_F);
                bool use_donor = (uniform_dist(rng) < current_Cr);
                T gene = use_donor ? donor : population[i][j];

                // GA-style mutation (adaptive)
                if (uniform_dist(rng) < adaptive_mut_rate) {
                    gene += normal_dist(rng) * adaptive_mut_sigma;
                }

                trials[i][j] = gene;
            }
            trial_fitness[i] = evaluate(trials[i]);
        }

        // Combine parents + trials, select top population_size
        std::vector<std::pair<T, std::vector<T>>> combined;
        combined.reserve(population_size * 2);
        for (int i = 0; i < population_size; ++i) {
            combined.push_back({fitness[i], population[i]});
            combined.push_back({trial_fitness[i], trials[i]});
        }
        std::sort(combined.begin(), combined.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        int num_elite = std::max(1, static_cast<int>(population_size * elite_fraction));
        for (int i = 0; i < population_size; ++i) {
            population[i] = combined[i].second;
            fitness[i] = combined[i].first;
        }
        // Ensure elites are preserved
        for (int i = 0; i < num_elite; ++i) {
            population[i] = combined[i].second;
            fitness[i] = combined[i].first;
        }

        // Diversity injection: refresh a few random individuals periodically
        if (it > 0 && it % 20 == 0) {
            int num_inject = std::max(1, population_size / 20);
            for (int k = 0; k < num_inject; ++k) {
                int idx = population_size - 1 - k;
                for (int j = 0; j < num_settings; ++j) {
                    population[idx][j] = mean[j] + sigma[j] * normal_dist(rng);
                }
                fitness[idx] = evaluate(population[idx]);
            }
        }

        // Update ES mean/sigma from top mu
        std::vector<T> new_mean(num_settings, static_cast<T>(0));
        for (int k = 0; k < mu; ++k) {
            const auto& x = population[k];
            for (int j = 0; j < num_settings; ++j) {
                new_mean[j] += weights[k] * x[j];
            }
        }
        // Feature: gently pull mean toward best individual for stability
        {
            const T mean_pull = static_cast<T>(0.2);
            const auto& best_x = population[0];
            for (int j = 0; j < num_settings; ++j) {
                new_mean[j] =
                    (static_cast<T>(1.0) - mean_pull) * new_mean[j] + mean_pull * best_x[j];
            }
        }
        std::vector<T> new_sigma(num_settings, static_cast<T>(0));
        for (int k = 0; k < mu; ++k) {
            const auto& x = population[k];
            for (int j = 0; j < num_settings; ++j) {
                T diff = x[j] - new_mean[j];
                new_sigma[j] += weights[k] * diff * diff;
            }
        }
        for (int j = 0; j < num_settings; ++j) {
            new_sigma[j] = std::sqrt(new_sigma[j]);
            new_sigma[j] = std::clamp(new_sigma[j], sigma_min, sigma_max);
        }

        mean.swap(new_mean);
        sigma.swap(new_sigma);

        auto t_it = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t_it_start)
                        .count() /
                    1000.0;
        if (it % 5 == 0) {
            std::cout << "Iter " << it << ": Best Loss = " << result.best_loss
                      << ", Time = " << t_it << "s" << std::endl;
        }
    }

    return result;
}

}  // namespace adaptive_echo
