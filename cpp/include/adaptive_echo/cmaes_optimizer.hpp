#pragma once

/**
 * CR-FM-NES optimizer.
 *
 * This implements the rank-one covariance natural evolution strategy from
 * Nomura and Ono (2022) for the synth's bounded latent search space.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "adaptive_echo/constants.hpp"

namespace adaptive_echo {

inline constexpr int kDefaultCRFMNESPopulationSize = 32;
inline constexpr float kDefaultCRFMNESInitialSigma = 2.8f;

template <typename T>
struct CRFMNESOptions {
    std::vector<T> initial_settings;
};

namespace detail {

inline std::mt19937& get_crfmnes_rng() {
    static thread_local std::mt19937 rng(std::random_device{}() + 123);
    return rng;
}

template <typename T>
inline T sigmoid_to_unit_interval(T x) {
    x = std::clamp(x, static_cast<T>(-500), static_cast<T>(500));
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
}

template <typename T>
inline T unit_interval_to_logit(T x) {
    constexpr T epsilon = static_cast<T>(1e-6);
    x = std::clamp(x, epsilon, static_cast<T>(1) - epsilon);
    return std::log(x / (static_cast<T>(1) - x));
}

template <typename T>
inline T squared_norm(const std::vector<T>& values) {
    T sum = static_cast<T>(0);
    for (T value : values) {
        sum += value * value;
    }
    return sum;
}

template <typename T>
inline T norm(const std::vector<T>& values) {
    return std::sqrt(squared_norm(values));
}

template <typename T>
inline T dot(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    T sum = static_cast<T>(0);
    for (size_t i = 0; i < lhs.size(); ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

template <typename T>
inline T safe_reciprocal(T value, T epsilon) {
    if (std::abs(value) <= epsilon) {
        return static_cast<T>(1) / (value < static_cast<T>(0) ? -epsilon : epsilon);
    }
    return static_cast<T>(1) / value;
}

template <typename T, typename LossFn>
class has_compute_batch {
    template <typename U>
    static auto test(int)
        -> decltype(std::declval<U&>().compute_batch(
                        std::declval<const std::vector<std::vector<T>>&>()),
                    std::true_type {});

    template <typename>
    static auto test(...) -> std::false_type;

   public:
    static constexpr bool value = decltype(test<LossFn>(0))::value;
};

template <typename T>
inline T get_h_inv(int dim) {
    auto f = [dim](T a) {
        return ((static_cast<T>(1) + a * a) * std::exp(a * a / static_cast<T>(2)) /
                    static_cast<T>(0.24)) -
               static_cast<T>(10) - static_cast<T>(dim);
    };
    auto f_prime = [](T a) {
        return (a * std::exp(a * a / static_cast<T>(2)) *
                (static_cast<T>(3) + a * a)) /
               static_cast<T>(0.24);
    };

    T h_inv = static_cast<T>(6);
    for (int i = 0; i < 64; ++i) {
        T value = f(h_inv);
        if (std::abs(value) <= static_cast<T>(1e-10)) {
            break;
        }
        T derivative = f_prime(h_inv);
        if (std::abs(derivative) <= static_cast<T>(1e-16)) {
            break;
        }
        T next = h_inv - static_cast<T>(0.5) * (value / derivative);
        if (std::abs(next - h_inv) <= static_cast<T>(1e-16)) {
            h_inv = next;
            break;
        }
        h_inv = next;
    }
    return h_inv;
}

template <typename T>
inline std::vector<int> sort_indices_by_fitness(const std::vector<T>& fitness) {
    std::vector<int> indices(fitness.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int lhs, int rhs) { return fitness[lhs] < fitness[rhs]; });
    return indices;
}

template <typename T, typename LossFn, typename SynthFn>
inline std::vector<T> evaluate_population(
    LossFn& loss_fn, SynthFn synth_fn, const std::vector<T>& time,
    const std::vector<std::vector<T>>& population_logits) {
    const size_t lambda = population_logits.size();
    const size_t num_settings = lambda > 0 ? population_logits.front().size() : 0;

    std::vector<std::vector<T>> generated(lambda);
    for (size_t i = 0; i < lambda; ++i) {
        std::vector<T> settings(num_settings);
        for (size_t j = 0; j < num_settings; ++j) {
            settings[j] = sigmoid_to_unit_interval(population_logits[i][j]);
        }
        generated[i] = synth_fn(settings, time);
    }

    if constexpr (has_compute_batch<T, LossFn>::value) {
        return loss_fn.compute_batch(generated);
    } else {
        std::vector<T> fitness(lambda);
        for (size_t i = 0; i < lambda; ++i) {
            fitness[i] = loss_fn(generated[i]);
        }
        return fitness;
    }
}

}  // namespace detail

template <typename T>
struct CRFMNESResult {
    std::vector<T> best_settings;
    T best_loss;
    int iterations_completed = 0;
    T final_sigma = static_cast<T>(0);
    int final_eval_count = 0;
};

template <typename T>
struct CRFMNESProgress {
    int generation = 0;
    T best_loss = std::numeric_limits<T>::max();
    T sigma = static_cast<T>(0);
    int eval_count = 0;
    T elapsed_seconds = static_cast<T>(0);
};

template <typename T, typename LossFn, typename SynthFn>
inline CRFMNESResult<T> run_crfmnes_optimization(
    LossFn& loss_fn, const std::vector<T>& time, SynthFn synth_fn, int lambda = -1,
    T initial_sigma = static_cast<T>(kDefaultCRFMNESInitialSigma),
    T time_limit = static_cast<T>(30),
    int max_iterations = 10000, bool verbose = true,
    std::function<void(const CRFMNESProgress<T>&)> progress_callback = {},
    const CRFMNESOptions<T>& options = {}) {
    const int dim = !options.initial_settings.empty()
                        ? static_cast<int>(options.initial_settings.size())
                        : adaptive_echo::constants::NUM_SETTINGS;
    auto t_start = std::chrono::steady_clock::now();
    auto& rng = detail::get_crfmnes_rng();

    if (lambda < 0) {
        lambda = kDefaultCRFMNESPopulationSize;
    }
    lambda = std::max(lambda, kDefaultCRFMNESPopulationSize);
    if ((lambda % 2) != 0) {
        ++lambda;
    }

    initial_sigma = std::max(initial_sigma, static_cast<T>(1e-6));

    std::vector<T> mean(dim, static_cast<T>(0));
    if (!options.initial_settings.empty()) {
        for (int i = 0; i < dim; ++i) {
            mean[i] = detail::unit_interval_to_logit(options.initial_settings[static_cast<size_t>(i)]);
        }
    }
    std::vector<T> diag_d(dim, static_cast<T>(1));
    std::vector<T> pc(dim, static_cast<T>(0));
    std::vector<T> ps(dim, static_cast<T>(0));
    std::vector<T> v(dim);

    {
        std::normal_distribution<T> init_dist(static_cast<T>(0),
                                              static_cast<T>(1) / std::sqrt(static_cast<T>(dim)));
        for (int i = 0; i < dim; ++i) {
            v[i] = init_dist(rng);
        }
    }

    std::vector<T> w_rank_hat(lambda);
    for (int i = 0; i < lambda; ++i) {
        const T raw = std::log(static_cast<T>(lambda / 2 + 1)) -
                      std::log(static_cast<T>(i + 1));
        w_rank_hat[i] = std::max(raw, static_cast<T>(0));
    }

    T w_rank_hat_sum = std::accumulate(w_rank_hat.begin(), w_rank_hat.end(), static_cast<T>(0));
    std::vector<T> w_rank(lambda);
    for (int i = 0; i < lambda; ++i) {
        w_rank[i] = w_rank_hat[i] / w_rank_hat_sum - static_cast<T>(1) / static_cast<T>(lambda);
    }

    T mueff_denom = static_cast<T>(0);
    for (int i = 0; i < lambda; ++i) {
        const T shifted = w_rank[i] + static_cast<T>(1) / static_cast<T>(lambda);
        mueff_denom += shifted * shifted;
    }
    const T mueff = static_cast<T>(1) / mueff_denom;

    const T cs =
        (mueff + static_cast<T>(2)) / (static_cast<T>(dim) + mueff + static_cast<T>(5));
    const T cc = (static_cast<T>(4) + mueff / static_cast<T>(dim)) /
                 (static_cast<T>(dim) + static_cast<T>(4) +
                  static_cast<T>(2) * mueff / static_cast<T>(dim));
    const T c1_cma =
        static_cast<T>(2) /
        (std::pow(static_cast<T>(dim) + static_cast<T>(1.3), static_cast<T>(2)) + mueff);
    const T chi_n = std::sqrt(static_cast<T>(dim)) *
                    (static_cast<T>(1) - static_cast<T>(1) / (static_cast<T>(4) *
                                                               static_cast<T>(dim)) +
                     static_cast<T>(1) /
                         (static_cast<T>(21) * static_cast<T>(dim) * static_cast<T>(dim)));
    const T h_inv = detail::get_h_inv<T>(dim);

    auto alpha_dist = [&](int feasible_count) {
        return h_inv * std::min(static_cast<T>(1),
                                std::sqrt(static_cast<T>(lambda) / static_cast<T>(dim))) *
               std::sqrt(static_cast<T>(feasible_count) / static_cast<T>(lambda));
    };
    auto eta_stag_sigma = [&](int feasible_count) {
        return std::tanh((static_cast<T>(0.024) * static_cast<T>(feasible_count) +
                          static_cast<T>(0.7) * static_cast<T>(dim) + static_cast<T>(20)) /
                         (static_cast<T>(dim) + static_cast<T>(12)));
    };
    auto eta_conv_sigma = [&](int feasible_count) {
        return static_cast<T>(2) *
               std::tanh((static_cast<T>(0.025) * static_cast<T>(feasible_count) +
                          static_cast<T>(0.75) * static_cast<T>(dim) + static_cast<T>(10)) /
                         (static_cast<T>(dim) + static_cast<T>(4)));
    };
    auto c1 = [&](int feasible_count) {
        return c1_cma * static_cast<T>(dim - 5) / static_cast<T>(6) *
               (static_cast<T>(feasible_count) / static_cast<T>(lambda));
    };
    auto eta_b = [&](int feasible_count) {
        return std::tanh((std::min(static_cast<T>(0.02) * static_cast<T>(feasible_count),
                                   static_cast<T>(3) * std::log(static_cast<T>(dim))) +
                          static_cast<T>(5)) /
                         (static_cast<T>(0.23) * static_cast<T>(dim) + static_cast<T>(25)));
    };

    T sigma = initial_sigma;
    const T eta_m = static_cast<T>(1);
    const T eta_move_sigma = static_cast<T>(1);
    const T min_d = static_cast<T>(1e-12);
    const T min_sigma = static_cast<T>(1e-8);
    const T max_sigma = static_cast<T>(20);
    const T epsilon = static_cast<T>(1e-12);

    std::normal_distribution<T> standard_normal(static_cast<T>(0), static_cast<T>(1));

    std::vector<std::vector<T>> z(lambda, std::vector<T>(dim, static_cast<T>(0)));
    std::vector<std::vector<T>> y(lambda, std::vector<T>(dim, static_cast<T>(0)));
    std::vector<std::vector<T>> population(lambda, std::vector<T>(dim, static_cast<T>(0)));

    CRFMNESResult<T> result;
    result.best_loss = std::numeric_limits<T>::max();
    std::vector<T> best_logits(dim, static_cast<T>(0));
    int eval_count = 0;

    for (int generation = 0; generation < max_iterations; ++generation) {
        auto t_now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<T>>(t_now - t_start).count();
        if (time_limit > static_cast<T>(0) && elapsed > time_limit) {
            if (verbose) {
                std::cout << "Time limit reached. Stopping." << std::endl;
            }
            break;
        }

        const int half_lambda = lambda / 2;
        for (int i = 0; i < half_lambda; ++i) {
            for (int j = 0; j < dim; ++j) {
                const T sample = standard_normal(rng);
                z[i][j] = sample;
                z[i + half_lambda][j] = -sample;
            }
        }

        const T norm_v = detail::norm(v);
        const T safe_norm_v = std::max(norm_v, epsilon);
        const T norm_v2 = norm_v * norm_v;
        const T norm_v4 = norm_v2 * norm_v2;
        const T sqrt_one_plus_norm_v2 = std::sqrt(static_cast<T>(1) + norm_v2);
        const T y_scale = sqrt_one_plus_norm_v2 - static_cast<T>(1);

        std::vector<T> vbar(dim, static_cast<T>(0));
        for (int j = 0; j < dim; ++j) {
            vbar[j] = v[j] / safe_norm_v;
        }

        for (int i = 0; i < lambda; ++i) {
            const T projection = detail::dot(vbar, z[i]);
            for (int j = 0; j < dim; ++j) {
                y[i][j] = z[i][j] + y_scale * vbar[j] * projection;
                population[i][j] = mean[j] + sigma * y[i][j] * diag_d[j];
            }
        }

        std::vector<T> fitness =
            detail::evaluate_population<T>(loss_fn, synth_fn, time, population);
        eval_count += lambda;

        std::vector<int> sorted_indices = detail::sort_indices_by_fitness(fitness);
        const int best_index = sorted_indices.front();
        const T best_generation_loss = fitness[best_index];
        if (best_generation_loss < result.best_loss) {
            result.best_loss = best_generation_loss;
            best_logits = population[best_index];
            result.best_settings.resize(static_cast<size_t>(dim));
            for (int j = 0; j < dim; ++j) {
                result.best_settings[static_cast<size_t>(j)] =
                    detail::sigmoid_to_unit_interval(best_logits[j]);
            }
        }

        std::vector<std::vector<T>> sorted_z(lambda, std::vector<T>(dim));
        std::vector<std::vector<T>> sorted_y(lambda, std::vector<T>(dim));
        std::vector<std::vector<T>> sorted_population(lambda, std::vector<T>(dim));
        std::vector<T> sorted_fitness(lambda);
        for (int rank = 0; rank < lambda; ++rank) {
            const size_t sorted_rank = static_cast<size_t>(rank);
            const size_t src = static_cast<size_t>(sorted_indices[sorted_rank]);
            sorted_z[sorted_rank] = z[src];
            sorted_y[sorted_rank] = y[src];
            sorted_population[sorted_rank] = population[src];
            sorted_fitness[sorted_rank] = fitness[src];
        }
        z.swap(sorted_z);
        y.swap(sorted_y);
        population.swap(sorted_population);
        fitness.swap(sorted_fitness);

        int feasible_count = 0;
        for (T value : fitness) {
            if (std::isfinite(value)) {
                ++feasible_count;
            }
        }
        feasible_count = std::max(feasible_count, 1);

        std::vector<T> weighted_z(dim, static_cast<T>(0));
        for (int i = 0; i < lambda; ++i) {
            for (int j = 0; j < dim; ++j) {
                weighted_z[j] += z[i][j] * w_rank[i];
            }
        }

        const T ps_scale = std::sqrt(cs * (static_cast<T>(2) - cs) * mueff);
        for (int j = 0; j < dim; ++j) {
            ps[j] = (static_cast<T>(1) - cs) * ps[j] + ps_scale * weighted_z[j];
        }
        const T ps_norm = detail::norm(ps);

        std::vector<T> weights_dist(lambda);
        std::vector<T> weights(lambda);
        {
            T distance_weight_sum = static_cast<T>(0);
            const T distance_alpha = alpha_dist(feasible_count);
            for (int i = 0; i < lambda; ++i) {
                weights_dist[i] =
                    w_rank_hat[i] * std::exp(distance_alpha * detail::norm(z[i]));
                distance_weight_sum += weights_dist[i];
            }
            if (distance_weight_sum <= epsilon) {
                distance_weight_sum = static_cast<T>(1);
            }
            for (int i = 0; i < lambda; ++i) {
                weights_dist[i] = weights_dist[i] / distance_weight_sum -
                                  static_cast<T>(1) / static_cast<T>(lambda);
            }
        }

        const bool moving_phase = ps_norm >= chi_n;
        if (moving_phase) {
            weights = weights_dist;
        } else {
            weights = w_rank;
        }

        const T eta_sigma =
            moving_phase
                ? eta_move_sigma
                : (ps_norm >= static_cast<T>(0.1) * chi_n ? eta_stag_sigma(feasible_count)
                                                          : eta_conv_sigma(feasible_count));

        std::vector<T> wxm(dim, static_cast<T>(0));
        for (int i = 0; i < lambda; ++i) {
            for (int j = 0; j < dim; ++j) {
                wxm[j] += (population[i][j] - mean[j]) * weights[i];
            }
        }

        const T pc_scale = std::sqrt(cc * (static_cast<T>(2) - cc) * mueff) / sigma;
        for (int j = 0; j < dim; ++j) {
            pc[j] = (static_cast<T>(1) - cc) * pc[j] + pc_scale * wxm[j];
            mean[j] += eta_m * wxm[j];
        }

        std::vector<std::vector<T>> ex_y(lambda + 1, std::vector<T>(dim, static_cast<T>(0)));
        for (int i = 0; i < lambda; ++i) {
            ex_y[i] = y[i];
        }
        for (int j = 0; j < dim; ++j) {
            ex_y[lambda][j] = pc[j] / std::max(diag_d[j], min_d);
        }

        std::vector<T> vbarbar(dim, static_cast<T>(0));
        T max_vbarbar = static_cast<T>(0);
        for (int j = 0; j < dim; ++j) {
            vbarbar[j] = vbar[j] * vbar[j];
            max_vbarbar = std::max(max_vbarbar, vbarbar[j]);
        }

        const T gammav = static_cast<T>(1) + norm_v2;
        const T alpha_vd =
            std::min(static_cast<T>(1),
                     std::sqrt(norm_v4 + (static_cast<T>(2) * gammav - std::sqrt(gammav)) /
                                            std::max(max_vbarbar, epsilon)) /
                         (static_cast<T>(2) + norm_v2));
        const T b = -(static_cast<T>(1) - alpha_vd * alpha_vd) * norm_v4 / gammav +
                    static_cast<T>(2) * alpha_vd * alpha_vd;

        std::vector<T> h(dim, static_cast<T>(0));
        std::vector<T> inv_h(dim, static_cast<T>(0));
        std::vector<T> inv_h_vbarbar(dim, static_cast<T>(0));
        T vbarbar_inv_h_dot = static_cast<T>(0);
        for (int j = 0; j < dim; ++j) {
            h[j] = static_cast<T>(2) -
                   (b + static_cast<T>(2) * alpha_vd * alpha_vd) * vbarbar[j];
            inv_h[j] = detail::safe_reciprocal(h[j], epsilon);
            inv_h_vbarbar[j] = inv_h[j] * vbarbar[j];
            vbarbar_inv_h_dot += vbarbar[j] * inv_h_vbarbar[j];
        }

        std::vector<std::vector<T>> s(lambda + 1, std::vector<T>(dim, static_cast<T>(0)));
        std::vector<std::vector<T>> t(lambda + 1, std::vector<T>(dim, static_cast<T>(0)));
        std::vector<T> ip_vbart(lambda + 1, static_cast<T>(0));

        for (int i = 0; i < lambda + 1; ++i) {
            const T ip_yvbar = detail::dot(vbar, ex_y[i]);
            for (int j = 0; j < dim; ++j) {
                t[i][j] = ex_y[i][j] * ip_yvbar -
                          vbar[j] * (ip_yvbar * ip_yvbar + gammav) / static_cast<T>(2);
            }
            ip_vbart[i] = detail::dot(vbar, t[i]);
        }

        for (int i = 0; i < lambda + 1; ++i) {
            T ip_s_step2_inv = static_cast<T>(0);
            const T ip_yvbar = detail::dot(vbar, ex_y[i]);
            for (int j = 0; j < dim; ++j) {
                const T yy = ex_y[i][j] * ex_y[i][j];
                const T yvbar_ip = ex_y[i][j] * vbar[j] * ip_yvbar;
                const T s_step1 =
                    yy - norm_v2 / gammav * yvbar_ip - static_cast<T>(1);
                const T s_step2 =
                    s_step1 -
                    alpha_vd / gammav *
                        ((static_cast<T>(2) + norm_v2) * t[i][j] * vbar[j] -
                         norm_v2 * vbarbar[j] * ip_vbart[i]);
                s[i][j] = s_step2 * inv_h[j];
                ip_s_step2_inv += inv_h_vbarbar[j] * s_step2;
            }

            const T correction =
                b / (static_cast<T>(1) + b * vbarbar_inv_h_dot) * ip_s_step2_inv;
            for (int j = 0; j < dim; ++j) {
                s[i][j] -= inv_h_vbarbar[j] * correction;
            }
        }

        for (int i = 0; i < lambda + 1; ++i) {
            T ip_svbarbar = static_cast<T>(0);
            for (int j = 0; j < dim; ++j) {
                ip_svbarbar += vbarbar[j] * s[i][j];
            }
            for (int j = 0; j < dim; ++j) {
                t[i][j] -= alpha_vd *
                           ((static_cast<T>(2) + norm_v2) * s[i][j] * vbar[j] -
                            vbar[j] * ip_svbarbar);
            }
        }

        std::vector<T> exw(lambda + 1, static_cast<T>(0));
        const T eta_b_value = eta_b(feasible_count);
        for (int i = 0; i < lambda; ++i) {
            exw[i] = eta_b_value * weights[i];
        }
        exw[lambda] = c1(feasible_count);

        std::vector<T> delta_v(dim, static_cast<T>(0));
        std::vector<T> delta_d(dim, static_cast<T>(0));
        for (int i = 0; i < lambda + 1; ++i) {
            for (int j = 0; j < dim; ++j) {
                delta_v[j] += t[i][j] * exw[i];
                delta_d[j] += s[i][j] * exw[i];
            }
        }

        for (int j = 0; j < dim; ++j) {
            v[j] += delta_v[j] / safe_norm_v;
            diag_d[j] = std::max(diag_d[j] + delta_d[j] * diag_d[j], min_d);
        }

        const T det_scale =
            std::exp(std::accumulate(diag_d.begin(), diag_d.end(), static_cast<T>(0),
                                     [](T acc, T value) { return acc + std::log(value); }) /
                         static_cast<T>(dim) +
                     std::log(static_cast<T>(1) + detail::squared_norm(v)) /
                         (static_cast<T>(2) * static_cast<T>(dim)));
        for (int j = 0; j < dim; ++j) {
            diag_d[j] /= det_scale;
        }

        T g_sigma = static_cast<T>(0);
        for (int i = 0; i < lambda; ++i) {
            g_sigma += (detail::squared_norm(z[i]) - static_cast<T>(dim)) * weights[i];
        }
        g_sigma /= static_cast<T>(dim);
        sigma *= std::exp(eta_sigma * static_cast<T>(0.5) * g_sigma);
        sigma = std::clamp(sigma, min_sigma, max_sigma);

        result.iterations_completed = generation + 1;

        if (progress_callback) {
            progress_callback(CRFMNESProgress<T> {
                generation + 1,
                result.best_loss,
                sigma,
                eval_count,
                elapsed,
            });
        }

        if (verbose && generation % 10 == 0) {
            std::cout << "Gen " << generation << ": Best Loss = " << result.best_loss
                      << " | Sigma = " << std::fixed << std::setprecision(4) << sigma
                      << " | Evals = " << eval_count << " | Elapsed: " << std::setprecision(1)
                      << elapsed << "s" << std::endl;
        }
    }

    if (result.best_settings.empty()) {
        result.best_settings.resize(static_cast<size_t>(dim), static_cast<T>(0.5));
    }
    result.final_sigma = sigma;
    result.final_eval_count = eval_count;
    return result;
}

}  // namespace adaptive_echo
