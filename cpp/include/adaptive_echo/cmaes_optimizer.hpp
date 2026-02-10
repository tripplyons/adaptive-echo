#pragma once

/**
 * CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer
 *
 * A derivative-free optimization algorithm that adapts the covariance matrix
 * of the search distribution to learn problem structure and guide exploration.
 *
 * Key features:
 * - Adaptive covariance matrix learning from successful mutations
 * - Weighted recombination for mean update
 * - Cumulative step-size adaptation (CSA) for sigma control
 * - Elitism support (best solution always preserved)
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
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
inline std::mt19937& get_cmaes_rng() {
    static thread_local std::mt19937 rng(std::random_device{}() + 123);
    return rng;
}
}  // namespace detail

template <typename T>
inline T sigmoid_cmaes(T x) {
    x = std::clamp(x, static_cast<T>(-500), static_cast<T>(500));
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
}

template <typename T>
struct CMAESResult {
    std::vector<T> best_settings;
    T best_loss;
    int iterations_completed = 0;
    T final_sigma = static_cast<T>(0);
    int final_eval_count = 0;
};

/**
 * Matrix operations helper for covariance matrix
 */
template <typename T>
class CovarianceMatrix {
public:
    explicit CovarianceMatrix(int n) : dim_(n), data_(n * n, static_cast<T>(0)) {
        // Initialize as identity matrix
        for (int i = 0; i < n; ++i) {
            (*this)(i, i) = static_cast<T>(1.0);
        }
    }

    T& operator()(int row, int col) { return data_[row * dim_ + col]; }
    const T& operator()(int row, int col) const { return data_[row * dim_ + col]; }

    int dim() const { return dim_; }

    // Matrix-vector multiplication
    std::vector<T> multiply(const std::vector<T>& x) const {
        std::vector<T> result(dim_, static_cast<T>(0));
        for (int i = 0; i < dim_; ++i) {
            for (int j = 0; j < dim_; ++j) {
                result[i] += (*this)(i, j) * x[j];
            }
        }
        return result;
    }

    // Add outer product: C += alpha * v * v^T
    void add_outer_product(const std::vector<T>& v, T alpha) {
        for (int i = 0; i < dim_; ++i) {
            for (int j = 0; j < dim_; ++j) {
                (*this)(i, j) += alpha * v[i] * v[j];
            }
        }
    }

    // Symmetrize matrix
    void symmetrize() {
        for (int i = 0; i < dim_; ++i) {
            for (int j = i + 1; j < dim_; ++j) {
                T avg = ((*this)(i, j) + (*this)(j, i)) / static_cast<T>(2);
                (*this)(i, j) = avg;
                (*this)(j, i) = avg;
            }
        }
    }

    // Scale matrix
    void scale(T factor) {
        for (auto& val : data_) {
            val *= factor;
        }
    }

    // Get diagonal elements
    std::vector<T> diagonal() const {
        std::vector<T> diag(dim_);
        for (int i = 0; i < dim_; ++i) {
            diag[i] = (*this)(i, i);
        }
        return diag;
    }

    // Set diagonal elements
    void set_diagonal(const std::vector<T>& diag) {
        for (int i = 0; i < dim_; ++i) {
            (*this)(i, i) = diag[i];
        }
    }

    // Eigenvalue decomposition using power iteration (simplified)
    // Returns eigenvalues and eigenvectors for covariance update
    void compute_eigendecomposition(std::vector<T>& eigenvalues,
                                    std::vector<std::vector<T>>& eigenvectors) const {
        // Use simple approximation: assume near-diagonal dominance for efficiency
        eigenvalues.resize(dim_);
        eigenvectors.assign(dim_, std::vector<T>(dim_, static_cast<T>(0)));

        for (int i = 0; i < dim_; ++i) {
            // Approximate eigenvalue by diagonal element
            eigenvalues[i] = std::max((*this)(i, i), static_cast<T>(1e-10));
            eigenvectors[i][i] = static_cast<T>(1.0);
        }

        // Normalize eigenvalues
        T sum = std::accumulate(eigenvalues.begin(), eigenvalues.end(), static_cast<T>(0));
        if (sum > 0) {
            for (auto& ev : eigenvalues) {
                ev /= sum;
            }
        }
    }

    // Limit condition number by adding small value to diagonal
    void regularize(T min_eigenvalue = static_cast<T>(1e-12)) {
        for (int i = 0; i < dim_; ++i) {
            (*this)(i, i) = std::max((*this)(i, i), min_eigenvalue);
        }
    }

private:
    int dim_;
    std::vector<T> data_;
};

/**
 * CMA-ES Optimizer
 *
 * Implementation of the Covariance Matrix Adaptation Evolution Strategy
 * algorithm with cumulative step-size adaptation and weighted recombination.
 *
 * @tparam T Numeric type (float or double)
 * @tparam LossFn Loss function type (callable with audio vector, returns T)
 * @tparam SynthFn Synthesis function type (callable with settings and time)
 */
template <typename T, typename LossFn, typename SynthFn>
inline CMAESResult<T> run_cmaes_optimization(LossFn& loss_fn,
                                           const std::vector<T>& time,
                                           SynthFn synth_fn,
                                           int lambda = -1,
                                           T initial_sigma = static_cast<T>(1.0),
                                           T time_limit = static_cast<T>(30.0),
                                           int max_iterations = 10000,
                                           bool verbose = true) {
    const int num_settings = adaptive_echo::constants::NUM_SETTINGS;
    auto t_start = std::chrono::steady_clock::now();
    auto& rng = detail::get_cmaes_rng();

    // Default population size: 4 + floor(3 * log(n))
    if (lambda < 0) {
        lambda = 4 + static_cast<int>(3 * std::log(num_settings));
    }

    // Number of parents (mu) - typically lambda / 2
    int mu = lambda / 2;

    // Recombination weights (log-scale)
    std::vector<T> weights(mu);
    T weights_sum = static_cast<T>(0);
    T weights_squared_sum = static_cast<T>(0);

    for (int i = 0; i < mu; ++i) {
        // w_i = log(mu + 0.5) - log(i + 1)
        weights[i] = std::log(static_cast<T>(mu) + static_cast<T>(0.5)) -
                     std::log(static_cast<T>(i) + static_cast<T>(1.0));
        weights_sum += weights[i];
    }

    // Normalize weights
    for (int i = 0; i < mu; ++i) {
        weights[i] /= weights_sum;
        weights_squared_sum += weights[i] * weights[i];
    }

    // Effective variance selection mass
    T mu_eff = static_cast<T>(1.0) / weights_squared_sum;

    // Strategy parameters
    T cc = static_cast<T>(4.0) / (static_cast<T>(num_settings) + static_cast<T>(4.0));  // Cumulation for C
    T cs = (mu_eff + static_cast<T>(2.0)) /
           (num_settings + mu_eff + static_cast<T>(3.0));  // Cumulation for sigma
    T c1 = static_cast<T>(2.0) /
           ((num_settings + static_cast<T>(1.3)) * (num_settings + static_cast<T>(1.3)) +
            mu_eff);  // Learning rate for rank-one update
    T cmu = std::min(static_cast<T>(1.0) - c1,
                     static_cast<T>(2.0) * (mu_eff - static_cast<T>(2.0) + static_cast<T>(1.0) / mu_eff) /
                         ((num_settings + static_cast<T>(2.0)) * (num_settings + static_cast<T>(2.0)) +
                          mu_eff));  // Learning rate for rank-mu update
    T damps = static_cast<T>(1.0) + static_cast<T>(2.0) * std::max(static_cast<T>(0),
                      std::sqrt((mu_eff - static_cast<T>(1.0)) / (num_settings + static_cast<T>(1.0))) -
                          static_cast<T>(1.0)) +
              cs;  // Damping for sigma

    // Initialize mean (in search space - logits)
    std::vector<T> mean(num_settings, static_cast<T>(0.0));

    // Initialize covariance matrix
    CovarianceMatrix<T> C(num_settings);

    // Evolution paths
    std::vector<T> pc(num_settings, static_cast<T>(0.0));  // Path for covariance
    std::vector<T> ps(num_settings, static_cast<T>(0.0));  // Path for sigma

    // Step size
    T sigma = initial_sigma;

    // Population and fitness
    std::vector<std::vector<T>> population(lambda, std::vector<T>(num_settings));
    std::vector<T> fitness(lambda);
    std::vector<int> indices(lambda);

    // Best solution tracking (elitism)
    std::vector<T> best_solution(num_settings);
    T best_fitness = std::numeric_limits<T>::max();

    auto evaluate = [&](const std::vector<T>& individual) {
        std::vector<T> settings(num_settings);
        for (int j = 0; j < num_settings; ++j) {
            settings[j] = sigmoid_cmaes(individual[j]);
        }
        auto audio = synth_fn(settings, time);
        return loss_fn(audio);
    };

    // Generate multivariate normal sample using Cholesky-like approach
    auto generate_sample = [&](std::mt19937& thread_rng) {
        std::vector<T> sample(num_settings);
        std::normal_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(1.0));

        // Generate standard normal
        for (int i = 0; i < num_settings; ++i) {
            sample[i] = dist(thread_rng);
        }

        // Transform by covariance (simplified - use diagonal approximation)
        std::vector<T> result(num_settings);
        for (int i = 0; i < num_settings; ++i) {
            T variance = std::max(C(i, i), static_cast<T>(1e-10));
            result[i] = mean[i] + sigma * std::sqrt(variance) * sample[i];
        }

        return result;
    };

    CMAESResult<T> result;
    result.best_loss = std::numeric_limits<T>::max();
    int eval_count = 0;

    for (int generation = 0; generation < max_iterations; ++generation) {
        auto t_now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<T>>(t_now - t_start).count();
        if (time_limit > 0 && elapsed > time_limit) {
            if (verbose) {
                std::cout << "Time limit reached. Stopping." << std::endl;
            }
            break;
        }

        // Generate and evaluate population
        for (int i = 0; i < lambda; ++i) {
            population[i] = generate_sample(rng);
            fitness[i] = evaluate(population[i]);
            indices[i] = i;
            eval_count++;

            // Track best
            if (fitness[i] < best_fitness) {
                best_fitness = fitness[i];
                best_solution = population[i];
            }
        }

        // Sort by fitness (best first)
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return fitness[a] < fitness[b]; });

        // Save best solution (elitism)
        if (fitness[indices[0]] < result.best_loss) {
            result.best_loss = fitness[indices[0]];
            result.best_settings.resize(num_settings);
            for (int j = 0; j < num_settings; ++j) {
                result.best_settings[j] = sigmoid_cmaes(population[indices[0]][j]);
            }
        }

        // Compute new mean (weighted recombination)
        std::vector<T> old_mean = mean;
        std::fill(mean.begin(), mean.end(), static_cast<T>(0.0));

        for (int i = 0; i < mu; ++i) {
            const auto& parent = population[indices[i]];
            for (int j = 0; j < num_settings; ++j) {
                mean[j] += weights[i] * parent[j];
            }
        }

        // Update evolution path for sigma (cumulation)
        std::vector<T> diff_mean(num_settings);
        for (int j = 0; j < num_settings; ++j) {
            diff_mean[j] = (mean[j] - old_mean[j]) / sigma;
        }

        // Simplified: use diagonal covariance for path update
        for (int j = 0; j < num_settings; ++j) {
            T variance = std::max(C(j, j), static_cast<T>(1e-10));
            ps[j] = (static_cast<T>(1.0) - cs) * ps[j] +
                    std::sqrt(cs * (static_cast<T>(2.0) - cs) * mu_eff) * diff_mean[j] / std::sqrt(variance);
        }

        // Compute hsig (stalling check for pc update)
        T ps_norm_sq = static_cast<T>(0);
        for (int j = 0; j < num_settings; ++j) {
            ps_norm_sq += ps[j] * ps[j];
        }
        T expected_ps_norm = static_cast<T>(num_settings);
        T hsig = (ps_norm_sq / expected_ps_norm / static_cast<T>(1.0) -
                  static_cast<T>(1.0) / (static_cast<T>(1.0) - std::pow(static_cast<T>(1.0) - cs, static_cast<T>(2) * eval_count / lambda))) <
                         static_cast<T>(0.3)
                     ? static_cast<T>(1.0)
                     : static_cast<T>(0.0);

        // Update evolution path for covariance
        for (int j = 0; j < num_settings; ++j) {
            T variance = std::max(C(j, j), static_cast<T>(1e-10));
            pc[j] = (static_cast<T>(1.0) - cc) * pc[j] +
                    hsig * std::sqrt(cc * (static_cast<T>(2.0) - cc) * mu_eff) * diff_mean[j] / std::sqrt(variance);
        }

        // Rank-mu update for covariance matrix
        C.scale(static_cast<T>(1.0) - c1 - cmu);

        // Add rank-one update
        if (hsig > static_cast<T>(0.5)) {
            for (int i = 0; i < num_settings; ++i) {
                for (int j = 0; j < num_settings; ++j) {
                    C(i, j) += c1 * pc[i] * pc[j];
                }
            }
        }

        // Add rank-mu update (simplified - diagonal only for efficiency)
        for (int k = 0; k < mu; ++k) {
            std::vector<T> diff(num_settings);
            for (int j = 0; j < num_settings; ++j) {
                diff[j] = (population[indices[k]][j] - old_mean[j]) / sigma;
            }

            for (int j = 0; j < num_settings; ++j) {
                C(j, j) += cmu * weights[k] * diff[j] * diff[j];
            }
        }

        // Symmetrize and regularize
        C.symmetrize();
        C.regularize(static_cast<T>(1e-12));

        // Cumulative step-size adaptation (CSA)
        T ps_norm = static_cast<T>(0);
        for (int j = 0; j < num_settings; ++j) {
            ps_norm += ps[j] * ps[j];
        }
        ps_norm = std::sqrt(ps_norm);

        T expected_length = std::sqrt(static_cast<T>(num_settings));
        sigma *= std::exp((cs / damps) * (ps_norm / expected_length - static_cast<T>(1.0)));

        // Clamp sigma to reasonable bounds
        sigma = std::clamp(sigma, static_cast<T>(1e-6), static_cast<T>(10.0));

        result.iterations_completed = generation + 1;

        if (verbose && generation % 10 == 0) {
            std::cout << "Gen " << generation << ": Best Loss = " << result.best_loss
                      << " | Sigma = " << std::fixed << std::setprecision(4) << sigma
                      << " | Evals = " << eval_count
                      << " | Elapsed: " << std::setprecision(1) << elapsed << "s" << std::endl;
        }

        // Check convergence
        if (sigma < static_cast<T>(1e-5)) {
            if (verbose) {
                std::cout << "Converged (sigma < 1e-5). Stopping." << std::endl;
            }
            break;
        }
    }

    // Ensure we return the absolute best (elitism)
    if (best_fitness < result.best_loss) {
        result.best_loss = best_fitness;
        result.best_settings.resize(num_settings);
        for (int j = 0; j < num_settings; ++j) {
            result.best_settings[j] = sigmoid_cmaes(best_solution[j]);
        }
    }

    result.final_sigma = sigma;
    result.final_eval_count = eval_count;

    return result;
}

}  // namespace adaptive_echo
