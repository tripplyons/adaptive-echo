#include <juce_audio_formats/juce_audio_formats.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/engine.hpp"
#include "adaptive_echo/loss.hpp"
#include "adaptive_echo/synth.hpp"

namespace {

struct TrainingStageConfig {
    const char* name = "";
    std::vector<int> indices;
    float time_fraction = 1.0f;
    float radius = 0.25f;
};

struct OptimizationRunResult {
    std::vector<float> best_settings;
    float best_loss = std::numeric_limits<float>::max();
    int eval_count = 0;
    double elapsed_seconds = 0.0;
};

struct AudioData {
    std::vector<float> samples;
    double sample_rate = 0.0;
};

struct TPEHyperparameters {
    double gamma = 0.22;
    int latent_divisor = 3;
    int max_latent_dim = 8;
    int min_latent_dim = 3;
    int min_init_samples = 10;
    int init_samples_multiplier = 4;
    int candidate_count = 384;
    double local_noise_std = 0.14;
    float coarse_radius = 0.24f;
    float shape_radius = 0.16f;
    float refine_radius = 0.08f;
};

std::vector<int> all_indices() {
    std::vector<int> indices(adaptive_echo::constants::NUM_SETTINGS);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

std::vector<float> select_settings(const std::vector<float>& settings,
                                   const std::vector<int>& indices) {
    std::vector<float> subset;
    subset.reserve(indices.size());
    for (int index : indices) {
        subset.push_back(settings[static_cast<size_t>(index)]);
    }
    return subset;
}

std::vector<float> merge_settings(const std::vector<float>& base, const std::vector<int>& indices,
                                  const std::vector<float>& updates) {
    auto merged = base;
    const size_t count = std::min(indices.size(), updates.size());
    for (size_t i = 0; i < count; ++i) {
        merged[static_cast<size_t>(indices[i])] = updates[i];
    }
    return merged;
}

AudioData load_audio_file(const std::string& path) {
    juce::AudioFormatManager format_manager;
    format_manager.registerBasicFormats();

    juce::File file { juce::String(path) };
    if (!file.existsAsFile()) {
        throw std::runtime_error("Input file not found: " + path);
    }

    std::unique_ptr<juce::AudioFormatReader> reader(format_manager.createReaderFor(file));
    if (reader == nullptr) {
        throw std::runtime_error("Failed to decode audio file: " + path);
    }

    juce::AudioBuffer<float> buffer(static_cast<int>(reader->numChannels),
                                    static_cast<int>(reader->lengthInSamples));
    if (!reader->read(&buffer, 0, static_cast<int>(reader->lengthInSamples), 0, true, true)) {
        throw std::runtime_error("Failed to read audio file: " + path);
    }

    AudioData result;
    result.sample_rate = reader->sampleRate;
    result.samples.assign(static_cast<size_t>(reader->lengthInSamples), 0.0f);
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
        float mixed = 0.0f;
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
            mixed += buffer.getSample(channel, sample);
        }
        result.samples[static_cast<size_t>(sample)] =
            mixed / static_cast<float>(buffer.getNumChannels());
    }
    return result;
}

void write_wav_file(const std::string& path, const std::vector<float>& audio, double sample_rate) {
    juce::File file { juce::String(path) };
    if (auto parent = file.getParentDirectory(); parent != juce::File()) {
        parent.createDirectory();
    }

    juce::WavAudioFormat wav_format;
    std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
    if (stream == nullptr) {
        throw std::runtime_error("Failed to create output file: " + path);
    }

    std::unique_ptr<juce::AudioFormatWriter> writer(
        wav_format.createWriterFor(stream.get(), sample_rate, 1, 24, {}, 0));
    if (writer == nullptr) {
        throw std::runtime_error("Failed to create wav writer: " + path);
    }

    stream.release();

    juce::AudioBuffer<float> buffer(1, static_cast<int>(audio.size()));
    if (!audio.empty()) {
        std::copy(audio.begin(), audio.end(), buffer.getWritePointer(0));
    }
    if (!writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples())) {
        throw std::runtime_error("Failed to write wav file: " + path);
    }
}

double gaussian_log_pdf(double x, double mean, double stddev) {
    const double safe_stddev = std::max(stddev, 1e-4);
    const double z = (x - mean) / safe_stddev;
    return -std::log(safe_stddev) - 0.5 * z * z - 0.5 * std::log(2.0 * M_PI);
}

std::vector<std::vector<double>> make_embedding(size_t rows, size_t cols, std::mt19937& rng) {
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::vector<std::vector<double>> embedding(rows, std::vector<double>(cols, 0.0));
    for (size_t row = 0; row < rows; ++row) {
        double norm = 0.0;
        for (size_t col = 0; col < cols; ++col) {
            const double value = gaussian(rng);
            embedding[row][col] = value;
            norm += value * value;
        }
        norm = std::sqrt(std::max(norm, 1e-9));
        for (size_t col = 0; col < cols; ++col) {
            embedding[row][col] /= norm;
        }
    }
    return embedding;
}

std::vector<float> decode_latent(const std::vector<float>& seed,
                                 const std::vector<std::vector<double>>& embedding,
                                 const std::vector<double>& latent, float radius) {
    std::vector<float> decoded(seed);
    for (size_t row = 0; row < decoded.size(); ++row) {
        double delta = 0.0;
        for (size_t col = 0; col < latent.size(); ++col) {
            delta += embedding[row][col] * latent[col];
        }
        decoded[row] =
            std::clamp(static_cast<float>(seed[row] + radius * delta), 0.0f, 1.0f);
    }
    return decoded;
}

double sample_cauchy(std::mt19937& rng, double location, double scale) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = std::clamp(uniform(rng), 1e-6, 1.0 - 1e-6);
    return location + scale * std::tan(M_PI * (u - 0.5));
}

class TPEModel {
public:
    explicit TPEModel(double gamma) : gamma_(gamma) {}

    bool fit(const std::vector<std::vector<double>>& inputs, const std::vector<double>& values) {
        if (inputs.empty() || inputs.size() != values.size()) {
            return false;
        }

        const size_t dim = inputs.front().size();
        const size_t n = values.size();
        const size_t good_count =
            std::clamp(static_cast<size_t>(std::ceil(gamma_ * static_cast<double>(n))),
                       static_cast<size_t>(1), n);

        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t lhs, size_t rhs) { return values[lhs] < values[rhs]; });

        good_means_.assign(dim, 0.0);
        good_stds_.assign(dim, 0.28);
        bad_means_.assign(dim, 0.0);
        bad_stds_.assign(dim, 0.55);

        for (size_t d = 0; d < dim; ++d) {
            double good_mean = 0.0;
            double bad_mean = 0.0;
            for (size_t i = 0; i < good_count; ++i) {
                good_mean += inputs[order[i]][d];
            }
            for (size_t i = good_count; i < n; ++i) {
                bad_mean += inputs[order[i]][d];
            }
            good_mean /= static_cast<double>(good_count);
            bad_mean /= static_cast<double>(std::max<size_t>(1, n - good_count));

            double good_var = 0.0;
            double bad_var = 0.0;
            for (size_t i = 0; i < good_count; ++i) {
                const double diff = inputs[order[i]][d] - good_mean;
                good_var += diff * diff;
            }
            for (size_t i = good_count; i < n; ++i) {
                const double diff = inputs[order[i]][d] - bad_mean;
                bad_var += diff * diff;
            }

            good_means_[d] = good_mean;
            bad_means_[d] = bad_mean;
            good_stds_[d] =
                std::clamp(std::sqrt(good_var / static_cast<double>(good_count) + 1e-4), 0.06, 0.45);
            bad_stds_[d] = std::clamp(
                std::sqrt(bad_var / static_cast<double>(std::max<size_t>(1, n - good_count)) + 1e-4),
                0.10, 0.75);
        }

        best_point_ = inputs[order.front()];
        return true;
    }

    std::vector<double> sample_candidate(std::mt19937& rng) const {
        std::vector<double> candidate(best_point_.size(), 0.0);
        for (size_t d = 0; d < candidate.size(); ++d) {
            std::normal_distribution<double> distribution(good_means_[d], good_stds_[d]);
            candidate[d] = std::clamp(distribution(rng), -1.0, 1.0);
        }
        return candidate;
    }

    double score_candidate(const std::vector<double>& candidate) const {
        double good_log_density = 0.0;
        double bad_log_density = 0.0;
        for (size_t d = 0; d < candidate.size(); ++d) {
            good_log_density += gaussian_log_pdf(candidate[d], good_means_[d], good_stds_[d]);
            bad_log_density += gaussian_log_pdf(candidate[d], bad_means_[d], bad_stds_[d]);
        }
        return good_log_density - bad_log_density;
    }

private:
    double gamma_ = 0.20;
    std::vector<double> good_means_;
    std::vector<double> good_stds_;
    std::vector<double> bad_means_;
    std::vector<double> bad_stds_;
    std::vector<double> best_point_;
};

OptimizationRunResult run_bayesian_optimizer(const std::vector<float>& target_audio,
                                             double time_limit_seconds,
                                             const TPEHyperparameters& hyperparams = {}) {
    using Clock = std::chrono::steady_clock;

    const auto time = adaptive_echo::make_time_axis(adaptive_echo::constants::NUM_SECONDS,
                                                    adaptive_echo::constants::TRAINING_SAMPLE_RATE);
    adaptive_echo::LossFunction<float> full_loss(target_audio);

    const std::vector<TrainingStageConfig> stages = {
        {"coarse",
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 23, 24, 25, 26, 27, 28, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 44, 45, 46, 48, 50},
         0.35f,
         hyperparams.coarse_radius},
        {"shape",
         {10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32, 33, 34, 47, 49},
         0.25f,
         hyperparams.shape_radius},
        {"refine", all_indices(), 0.40f, hyperparams.refine_radius},
    };

    OptimizationRunResult result;
    result.best_settings = adaptive_echo::default_settings();
    auto current_settings = result.best_settings;

    const auto global_start = Clock::now();
    const auto total_fraction =
        std::accumulate(stages.begin(), stages.end(), 0.0f,
                        [](float sum, const TrainingStageConfig& stage) { return sum + stage.time_fraction; });

    std::mt19937 rng(1337);

    for (const auto& stage : stages) {
        const double elapsed_total =
            std::chrono::duration<double>(Clock::now() - global_start).count();
        const double remaining_time = std::max(0.0, time_limit_seconds - elapsed_total);
        if (remaining_time <= 0.0) {
            break;
        }

        const double stage_time_limit =
            std::max(1.0, remaining_time * (stage.time_fraction / total_fraction));
        const auto stage_seed = select_settings(current_settings, stage.indices);
        const size_t stage_dim = stage.indices.size();
        const size_t latent_dim =
            std::min<size_t>(static_cast<size_t>(hyperparams.max_latent_dim),
                             std::max<size_t>(static_cast<size_t>(hyperparams.min_latent_dim),
                                              stage_dim / static_cast<size_t>(
                                                              std::max(1, hyperparams.latent_divisor))));
        const auto embedding = make_embedding(stage_dim, latent_dim, rng);

        std::vector<std::vector<double>> latent_points;
        std::vector<double> losses;
        std::vector<float> best_stage_settings = stage_seed;
        double best_stage_loss = std::numeric_limits<double>::max();

        auto evaluate_candidate = [&](const std::vector<double>& latent) {
            const auto stage_settings = decode_latent(stage_seed, embedding, latent, stage.radius);
            const auto merged = merge_settings(current_settings, stage.indices, stage_settings);
            const auto rendered = adaptive_echo::synth(
                merged, time, static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
            const double loss = static_cast<double>(full_loss(rendered));

            latent_points.push_back(latent);
            losses.push_back(loss);
            ++result.eval_count;
            if (loss < best_stage_loss) {
                best_stage_loss = loss;
                best_stage_settings = stage_settings;
            }
        };

        const auto stage_start = Clock::now();
        evaluate_candidate(std::vector<double>(latent_dim, 0.0));

        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        const int init_samples =
            std::max(hyperparams.min_init_samples,
                     static_cast<int>(latent_dim) * hyperparams.init_samples_multiplier);
        for (int i = 1; i < init_samples; ++i) {
            std::vector<double> latent(latent_dim, 0.0);
            for (size_t dim = 0; dim < latent_dim; ++dim) {
                latent[dim] = uniform(rng);
            }
            evaluate_candidate(latent);
        }

        while (std::chrono::duration<double>(Clock::now() - stage_start).count() < stage_time_limit) {
            TPEModel tpe(hyperparams.gamma);
            const bool tpe_ready = tpe.fit(latent_points, losses);

            std::vector<double> best_candidate(latent_dim, 0.0);
            double best_acquisition = -1.0;
            std::normal_distribution<double> local_noise(0.0, hyperparams.local_noise_std);

            const auto best_latent_index = static_cast<size_t>(
                std::distance(losses.begin(), std::min_element(losses.begin(), losses.end())));
            const auto& incumbent = latent_points[best_latent_index];

            const int candidate_count = hyperparams.candidate_count;
            for (int i = 0; i < candidate_count; ++i) {
                std::vector<double> candidate(latent_dim, 0.0);
                if (tpe_ready && (i % 4) != 0) {
                    candidate = tpe.sample_candidate(rng);
                } else {
                    for (size_t dim = 0; dim < latent_dim; ++dim) {
                        double value = uniform(rng);
                        if ((i % 3) != 0) {
                            value = std::clamp(incumbent[dim] + local_noise(rng), -1.0, 1.0);
                        }
                        candidate[dim] = value;
                    }
                }

                const double acquisition =
                    tpe_ready ? tpe.score_candidate(candidate) : uniform(rng);

                if (acquisition > best_acquisition) {
                    best_acquisition = acquisition;
                    best_candidate = candidate;
                }
            }

            evaluate_candidate(best_candidate);
        }

        current_settings = merge_settings(current_settings, stage.indices, best_stage_settings);
        const auto rendered = adaptive_echo::synth(
            current_settings, time, static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const float full_stage_loss = full_loss(rendered);
        if (full_stage_loss < result.best_loss) {
            result.best_loss = full_stage_loss;
            result.best_settings = current_settings;
        }
    }

    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - global_start).count();
    if (result.best_loss == std::numeric_limits<float>::max()) {
        const auto rendered = adaptive_echo::synth(
            current_settings, time, static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        result.best_loss = full_loss(rendered);
        result.best_settings = current_settings;
    }
    return result;
}

OptimizationRunResult run_jade_optimizer(const std::vector<float>& target_audio,
                                         double time_limit_seconds) {
    using Clock = std::chrono::steady_clock;

    const auto time = adaptive_echo::make_time_axis(adaptive_echo::constants::NUM_SECONDS,
                                                    adaptive_echo::constants::TRAINING_SAMPLE_RATE);
    adaptive_echo::LossFunction<float> full_loss(target_audio);

    const std::vector<TrainingStageConfig> stages = {
        {"coarse",
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 23, 24, 25, 26, 27, 28, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 44, 45, 46, 48, 50},
         0.35f,
         0.20f},
        {"shape",
         {10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32, 33, 34, 47, 49},
         0.25f,
         0.14f},
        {"refine", all_indices(), 0.40f, 0.08f},
    };

    OptimizationRunResult result;
    result.best_settings = adaptive_echo::default_settings();
    auto current_settings = result.best_settings;

    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::uniform_real_distribution<double> unit_uniform(0.0, 1.0);
    std::normal_distribution<double> cr_noise(0.0, 0.10);

    const auto global_start = Clock::now();
    const auto total_fraction =
        std::accumulate(stages.begin(), stages.end(), 0.0f,
                        [](float sum, const TrainingStageConfig& stage) {
                            return sum + stage.time_fraction;
                        });

    for (const auto& stage : stages) {
        const double elapsed_total =
            std::chrono::duration<double>(Clock::now() - global_start).count();
        const double remaining_time = std::max(0.0, time_limit_seconds - elapsed_total);
        if (remaining_time <= 0.0) {
            break;
        }

        const double stage_time_limit =
            std::max(1.0, remaining_time * (stage.time_fraction / total_fraction));
        const auto stage_seed = select_settings(current_settings, stage.indices);
        const size_t stage_dim = stage.indices.size();
        const size_t latent_dim = std::min<size_t>(6, std::max<size_t>(3, stage_dim / 4));
        const auto embedding = make_embedding(stage_dim, latent_dim, rng);

        const int population_size = std::max(10, static_cast<int>(latent_dim) * 3);
        std::vector<std::vector<double>> population(static_cast<size_t>(population_size),
                                                    std::vector<double>(latent_dim, 0.0));
        std::vector<double> fitness(static_cast<size_t>(population_size),
                                    std::numeric_limits<double>::max());
        std::vector<std::vector<double>> archive;
        archive.reserve(static_cast<size_t>(population_size));

        auto evaluate_latent = [&](const std::vector<double>& latent) {
            const auto stage_settings = decode_latent(stage_seed, embedding, latent, stage.radius);
            const auto merged = merge_settings(current_settings, stage.indices, stage_settings);
            const auto rendered = adaptive_echo::synth(
                merged, time, static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
            ++result.eval_count;
            return std::pair<double, std::vector<float>>(
                static_cast<double>(full_loss(rendered)), stage_settings);
        };

        auto best_stage_settings = stage_seed;
        double best_stage_loss = std::numeric_limits<double>::max();

        for (int i = 0; i < population_size; ++i) {
            if (i != 0) {
                for (size_t d = 0; d < latent_dim; ++d) {
                    population[static_cast<size_t>(i)][d] = uniform(rng);
                }
            }
            auto [loss, stage_settings] = evaluate_latent(population[static_cast<size_t>(i)]);
            fitness[static_cast<size_t>(i)] = loss;
            if (loss < best_stage_loss) {
                best_stage_loss = loss;
                best_stage_settings = stage_settings;
            }
        }

        double mu_f = 0.48;
        double mu_cr = 0.72;
        const auto stage_start = Clock::now();

        while (std::chrono::duration<double>(Clock::now() - stage_start).count() < stage_time_limit) {
            std::vector<size_t> order(static_cast<size_t>(population_size));
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](size_t lhs, size_t rhs) { return fitness[lhs] < fitness[rhs]; });
            const int pbest_count = std::max(2, population_size / 5);

            std::vector<double> success_f;
            std::vector<double> success_cr;

            for (int i = 0; i < population_size; ++i) {
                double f = sample_cauchy(rng, mu_f, 0.10);
                while (f <= 0.0) {
                    f = sample_cauchy(rng, mu_f, 0.10);
                }
                f = std::min(f, 1.0);

                const double cr = std::clamp(mu_cr + cr_noise(rng), 0.0, 1.0);
                const size_t pbest_index = order[static_cast<size_t>(
                    std::uniform_int_distribution<int>(0, pbest_count - 1)(rng))];

                int r1 = i;
                while (r1 == i) {
                    r1 = std::uniform_int_distribution<int>(0, population_size - 1)(rng);
                }

                const int combined_size =
                    population_size + static_cast<int>(archive.size());
                int r2 = i;
                while (r2 == i || r2 == r1) {
                    r2 = std::uniform_int_distribution<int>(0, combined_size - 1)(rng);
                }

                const auto& x_i = population[static_cast<size_t>(i)];
                const auto& x_pbest = population[pbest_index];
                const auto& x_r1 = population[static_cast<size_t>(r1)];
                const auto& x_r2 =
                    r2 < population_size ? population[static_cast<size_t>(r2)]
                                         : archive[static_cast<size_t>(r2 - population_size)];

                std::vector<double> mutant(latent_dim, 0.0);
                std::vector<double> trial = x_i;
                const int j_rand = std::uniform_int_distribution<int>(
                    0, static_cast<int>(latent_dim) - 1)(rng);

                for (size_t d = 0; d < latent_dim; ++d) {
                    mutant[d] = x_i[d] + f * (x_pbest[d] - x_i[d]) + f * (x_r1[d] - x_r2[d]);
                    mutant[d] = std::clamp(mutant[d], -1.0, 1.0);
                    if (unit_uniform(rng) < cr || static_cast<int>(d) == j_rand) {
                        trial[d] = mutant[d];
                    }
                }

                auto [trial_loss, trial_settings] = evaluate_latent(trial);
                if (trial_loss <= fitness[static_cast<size_t>(i)]) {
                    archive.push_back(x_i);
                    if (archive.size() > population.size()) {
                        archive.erase(archive.begin() + std::uniform_int_distribution<int>(
                                                            0, static_cast<int>(archive.size()) - 1)(rng));
                    }
                    population[static_cast<size_t>(i)] = std::move(trial);
                    fitness[static_cast<size_t>(i)] = trial_loss;
                    success_f.push_back(f);
                    success_cr.push_back(cr);
                    if (trial_loss < best_stage_loss) {
                        best_stage_loss = trial_loss;
                        best_stage_settings = trial_settings;
                    }
                }

                if (std::chrono::duration<double>(Clock::now() - stage_start).count() >=
                    stage_time_limit) {
                    break;
                }
            }

            if (!success_f.empty()) {
                double numer = 0.0;
                double denom = 0.0;
                for (double value : success_f) {
                    numer += value * value;
                    denom += value;
                }
                if (denom > 1e-8) {
                    mu_f = 0.9 * mu_f + 0.1 * (numer / denom);
                }
                const double mean_cr =
                    std::accumulate(success_cr.begin(), success_cr.end(), 0.0) /
                    static_cast<double>(success_cr.size());
                mu_cr = 0.9 * mu_cr + 0.1 * mean_cr;
            }
        }

        current_settings = merge_settings(current_settings, stage.indices, best_stage_settings);
        const auto rendered = adaptive_echo::synth(
            current_settings, time,
            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const float full_stage_loss = full_loss(rendered);
        if (full_stage_loss < result.best_loss) {
            result.best_loss = full_stage_loss;
            result.best_settings = current_settings;
        }
    }

    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - global_start).count();
    if (result.best_loss >= std::numeric_limits<float>::max() * 0.5f) {
        const auto rendered = adaptive_echo::synth(
            current_settings, time,
            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        result.best_loss = full_loss(rendered);
        result.best_settings = current_settings;
    }
    return result;
}

OptimizationRunResult run_lshade_optimizer(const std::vector<float>& target_audio,
                                           double time_limit_seconds) {
    using Clock = std::chrono::steady_clock;

    const auto time = adaptive_echo::make_time_axis(adaptive_echo::constants::NUM_SECONDS,
                                                    adaptive_echo::constants::TRAINING_SAMPLE_RATE);
    adaptive_echo::LossFunction<float> full_loss(target_audio);

    const std::vector<TrainingStageConfig> stages = {
        {"coarse",
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 23, 24, 25, 26, 27, 28, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 44, 45, 46, 48, 50},
         0.35f,
         0.22f},
        {"shape",
         {10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32, 33, 34, 47, 49},
         0.25f,
         0.14f},
        {"refine", all_indices(), 0.40f, 0.08f},
    };

    OptimizationRunResult result;
    result.best_settings = adaptive_echo::default_settings();
    auto current_settings = result.best_settings;

    std::mt19937 rng(4040);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::uniform_real_distribution<double> unit_uniform(0.0, 1.0);
    std::normal_distribution<double> cr_noise(0.0, 0.10);

    const auto global_start = Clock::now();
    const auto total_fraction =
        std::accumulate(stages.begin(), stages.end(), 0.0f,
                        [](float sum, const TrainingStageConfig& stage) {
                            return sum + stage.time_fraction;
                        });

    for (const auto& stage : stages) {
        const double elapsed_total =
            std::chrono::duration<double>(Clock::now() - global_start).count();
        const double remaining_time = std::max(0.0, time_limit_seconds - elapsed_total);
        if (remaining_time <= 0.0) {
            break;
        }

        const double stage_time_limit =
            std::max(1.0, remaining_time * (stage.time_fraction / total_fraction));
        const auto stage_seed = select_settings(current_settings, stage.indices);
        const size_t stage_dim = stage.indices.size();
        const size_t latent_dim = std::min<size_t>(6, std::max<size_t>(3, stage_dim / 4));
        const auto embedding = make_embedding(stage_dim, latent_dim, rng);

        const int initial_population = std::max(14, static_cast<int>(latent_dim) * 5);
        const int min_population = std::max(6, static_cast<int>(latent_dim) * 2);
        std::vector<std::vector<double>> population(static_cast<size_t>(initial_population),
                                                    std::vector<double>(latent_dim, 0.0));
        std::vector<double> fitness(static_cast<size_t>(initial_population),
                                    std::numeric_limits<double>::max());
        std::vector<std::vector<double>> archive;
        archive.reserve(static_cast<size_t>(initial_population));

        auto evaluate_latent = [&](const std::vector<double>& latent) {
            const auto stage_settings = decode_latent(stage_seed, embedding, latent, stage.radius);
            const auto merged = merge_settings(current_settings, stage.indices, stage_settings);
            const auto rendered = adaptive_echo::synth(
                merged, time, static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
            ++result.eval_count;
            return std::pair<double, std::vector<float>>(
                static_cast<double>(full_loss(rendered)), stage_settings);
        };

        auto best_stage_settings = stage_seed;
        double best_stage_loss = std::numeric_limits<double>::max();

        for (int i = 0; i < initial_population; ++i) {
            if (i != 0) {
                for (size_t d = 0; d < latent_dim; ++d) {
                    population[static_cast<size_t>(i)][d] = uniform(rng);
                }
            }
            auto [loss, stage_settings] = evaluate_latent(population[static_cast<size_t>(i)]);
            fitness[static_cast<size_t>(i)] = loss;
            if (loss < best_stage_loss) {
                best_stage_loss = loss;
                best_stage_settings = stage_settings;
            }
        }

        constexpr int memory_size = 6;
        std::vector<double> memory_f(memory_size, 0.55);
        std::vector<double> memory_cr(memory_size, 0.65);
        int memory_index = 0;

        const auto stage_start = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - stage_start).count() < stage_time_limit) {
            const double stage_progress =
                std::clamp(std::chrono::duration<double>(Clock::now() - stage_start).count() /
                               stage_time_limit,
                           0.0, 1.0);
            const int current_population = std::max(
                min_population,
                static_cast<int>(std::round(initial_population -
                                            stage_progress * (initial_population - min_population))));

            if (static_cast<int>(population.size()) > current_population) {
                std::vector<size_t> order(population.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                          [&](size_t lhs, size_t rhs) { return fitness[lhs] < fitness[rhs]; });

                std::vector<std::vector<double>> reduced_population;
                std::vector<double> reduced_fitness;
                reduced_population.reserve(static_cast<size_t>(current_population));
                reduced_fitness.reserve(static_cast<size_t>(current_population));
                for (int i = 0; i < current_population; ++i) {
                    reduced_population.push_back(population[order[static_cast<size_t>(i)]]);
                    reduced_fitness.push_back(fitness[order[static_cast<size_t>(i)]]);
                }
                population = std::move(reduced_population);
                fitness = std::move(reduced_fitness);
                if (archive.size() > population.size()) {
                    archive.resize(population.size());
                }
            }

            std::vector<size_t> order(population.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](size_t lhs, size_t rhs) { return fitness[lhs] < fitness[rhs]; });
            const int pbest_count = std::max(2, static_cast<int>(population.size()) / 6);

            std::vector<double> success_f;
            std::vector<double> success_cr;
            std::vector<double> success_delta;

            for (size_t i = 0; i < population.size(); ++i) {
                const int mem_slot =
                    std::uniform_int_distribution<int>(0, memory_size - 1)(rng);
                double f = sample_cauchy(rng, memory_f[static_cast<size_t>(mem_slot)], 0.10);
                while (f <= 0.0) {
                    f = sample_cauchy(rng, memory_f[static_cast<size_t>(mem_slot)], 0.10);
                }
                f = std::min(f, 1.0);

                const double cr = std::clamp(memory_cr[static_cast<size_t>(mem_slot)] + cr_noise(rng),
                                             0.0, 1.0);

                const size_t pbest_index = order[static_cast<size_t>(
                    std::uniform_int_distribution<int>(0, pbest_count - 1)(rng))];

                int r1 = static_cast<int>(i);
                while (r1 == static_cast<int>(i)) {
                    r1 = std::uniform_int_distribution<int>(0, static_cast<int>(population.size()) - 1)(rng);
                }

                const int combined_size =
                    static_cast<int>(population.size()) + static_cast<int>(archive.size());
                int r2 = static_cast<int>(i);
                while (r2 == static_cast<int>(i) || r2 == r1) {
                    r2 = std::uniform_int_distribution<int>(0, combined_size - 1)(rng);
                }

                const auto& x_i = population[i];
                const auto& x_pbest = population[pbest_index];
                const auto& x_r1 = population[static_cast<size_t>(r1)];
                const auto& x_r2 =
                    r2 < static_cast<int>(population.size())
                        ? population[static_cast<size_t>(r2)]
                        : archive[static_cast<size_t>(r2 - static_cast<int>(population.size()))];

                std::vector<double> mutant(latent_dim, 0.0);
                std::vector<double> trial = x_i;
                const int j_rand = std::uniform_int_distribution<int>(
                    0, static_cast<int>(latent_dim) - 1)(rng);

                for (size_t d = 0; d < latent_dim; ++d) {
                    mutant[d] = x_i[d] + f * (x_pbest[d] - x_i[d]) + f * (x_r1[d] - x_r2[d]);
                    mutant[d] = std::clamp(mutant[d], -1.0, 1.0);
                    if (unit_uniform(rng) < cr || static_cast<int>(d) == j_rand) {
                        trial[d] = mutant[d];
                    }
                }

                auto [trial_loss, trial_settings] = evaluate_latent(trial);
                if (trial_loss <= fitness[i]) {
                    const double improvement = std::max(1e-8, fitness[i] - trial_loss);
                    archive.push_back(x_i);
                    if (archive.size() > population.size()) {
                        archive.erase(archive.begin() + std::uniform_int_distribution<int>(
                                                            0, static_cast<int>(archive.size()) - 1)(rng));
                    }
                    population[i] = std::move(trial);
                    fitness[i] = trial_loss;
                    success_f.push_back(f);
                    success_cr.push_back(cr);
                    success_delta.push_back(improvement);
                    if (trial_loss < best_stage_loss) {
                        best_stage_loss = trial_loss;
                        best_stage_settings = trial_settings;
                    }
                }

                if (std::chrono::duration<double>(Clock::now() - stage_start).count() >=
                    stage_time_limit) {
                    break;
                }
            }

            if (!success_f.empty()) {
                double weight_sum = std::accumulate(success_delta.begin(), success_delta.end(), 0.0);
                double weighted_f_num = 0.0;
                double weighted_f_den = 0.0;
                double weighted_cr = 0.0;
                for (size_t i = 0; i < success_f.size(); ++i) {
                    const double w = success_delta[i] / weight_sum;
                    weighted_f_num += w * success_f[i] * success_f[i];
                    weighted_f_den += w * success_f[i];
                    weighted_cr += w * success_cr[i];
                }
                if (weighted_f_den > 1e-8) {
                    memory_f[static_cast<size_t>(memory_index)] = weighted_f_num / weighted_f_den;
                }
                memory_cr[static_cast<size_t>(memory_index)] = weighted_cr;
                memory_index = (memory_index + 1) % memory_size;
            }
        }

        current_settings = merge_settings(current_settings, stage.indices, best_stage_settings);
        const auto rendered = adaptive_echo::synth(
            current_settings, time,
            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const float full_stage_loss = full_loss(rendered);
        if (full_stage_loss < result.best_loss) {
            result.best_loss = full_stage_loss;
            result.best_settings = current_settings;
        }
    }

    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - global_start).count();
    if (result.best_loss >= std::numeric_limits<float>::max() * 0.5f) {
        const auto rendered = adaptive_echo::synth(
            current_settings, time,
            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        result.best_loss = full_loss(rendered);
        result.best_settings = current_settings;
    }
    return result;
}

OptimizationRunResult run_hybrid_optimizer(const std::vector<float>& target_audio,
                                           double time_limit_seconds) {
    const double tpe_budget = std::max(2.5, time_limit_seconds * 0.55);
    auto tpe_result = run_bayesian_optimizer(target_audio, tpe_budget);

    using Clock = std::chrono::steady_clock;
    const auto time = adaptive_echo::make_time_axis(adaptive_echo::constants::NUM_SECONDS,
                                                    adaptive_echo::constants::TRAINING_SAMPLE_RATE);
    adaptive_echo::LossFunction<float> full_loss(target_audio);

    OptimizationRunResult result = tpe_result;
    auto current_settings = tpe_result.best_settings;
    std::mt19937 rng(5050);
    std::uniform_real_distribution<double> unit_uniform(0.0, 1.0);
    std::normal_distribution<double> local_noise(0.0, 0.08);

    const auto start = Clock::now();
    const double remaining_budget = std::max(2.0, time_limit_seconds - tpe_result.elapsed_seconds);
    std::vector<std::vector<double>> population(14, std::vector<double>(6, 0.0));
    std::vector<double> fitness(14, std::numeric_limits<double>::max());
    const auto embedding = make_embedding(adaptive_echo::constants::NUM_SETTINGS, 6, rng);

    auto evaluate_latent = [&](const std::vector<double>& latent) {
        const auto stage_settings = decode_latent(current_settings, embedding, latent, 0.06f);
        const auto rendered = adaptive_echo::synth(
            stage_settings, time,
            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        ++result.eval_count;
        return std::pair<double, std::vector<float>>(static_cast<double>(full_loss(rendered)),
                                                     stage_settings);
    };

    for (size_t i = 0; i < population.size(); ++i) {
        for (double& value : population[i]) {
            value = std::clamp(local_noise(rng), -1.0, 1.0);
        }
        auto [loss, settings] = evaluate_latent(population[i]);
        fitness[i] = loss;
        if (loss < result.best_loss) {
            result.best_loss = static_cast<float>(loss);
            result.best_settings = settings;
        }
    }

    while (std::chrono::duration<double>(Clock::now() - start).count() < remaining_budget) {
        std::vector<size_t> order(population.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t lhs, size_t rhs) { return fitness[lhs] < fitness[rhs]; });

        for (size_t i = 0; i < population.size(); ++i) {
            const auto& best = population[order[0]];
            const auto& pbest = population[order[std::min<size_t>(2, order.size() - 1)]];
            std::vector<double> trial = population[i];
            for (size_t d = 0; d < trial.size(); ++d) {
                const double donor =
                    population[i][d] + 0.55 * (best[d] - population[i][d]) +
                    0.25 * (pbest[d] - population[i][d]) + local_noise(rng) * 0.35;
                if (unit_uniform(rng) < 0.8) {
                    trial[d] = std::clamp(donor, -1.0, 1.0);
                }
            }

            auto [loss, settings] = evaluate_latent(trial);
            if (loss <= fitness[i]) {
                population[i] = std::move(trial);
                fitness[i] = loss;
                if (loss < result.best_loss) {
                    result.best_loss = static_cast<float>(loss);
                    result.best_settings = settings;
                }
            }

            if (std::chrono::duration<double>(Clock::now() - start).count() >= remaining_budget) {
                break;
            }
        }
    }

    result.elapsed_seconds = tpe_result.elapsed_seconds +
                             std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

OptimizationRunResult run_simulated_annealing_optimizer(const std::vector<float>& target_audio,
                                                        double time_limit_seconds) {
    using Clock = std::chrono::steady_clock;

    const auto time = adaptive_echo::make_time_axis(adaptive_echo::constants::NUM_SECONDS,
                                                    adaptive_echo::constants::TRAINING_SAMPLE_RATE);
    adaptive_echo::LossFunction<float> full_loss(target_audio);

    const std::vector<TrainingStageConfig> stages = {
        {"coarse",
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 23, 24, 25, 26, 27, 28, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 44, 45, 46, 48, 50},
         0.35f,
         0.20f},
        {"shape",
         {10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32, 33, 34, 47, 49},
         0.25f,
         0.12f},
        {"refine", all_indices(), 0.40f, 0.06f},
    };

    OptimizationRunResult result;
    result.best_settings = adaptive_echo::default_settings();
    auto current_settings = result.best_settings;

    std::mt19937 rng(6060);
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::uniform_real_distribution<double> unit_uniform(0.0, 1.0);

    const auto global_start = Clock::now();
    const auto total_fraction =
        std::accumulate(stages.begin(), stages.end(), 0.0f,
                        [](float sum, const TrainingStageConfig& stage) {
                            return sum + stage.time_fraction;
                        });

    for (const auto& stage : stages) {
        const double elapsed_total =
            std::chrono::duration<double>(Clock::now() - global_start).count();
        const double remaining_time = std::max(0.0, time_limit_seconds - elapsed_total);
        if (remaining_time <= 0.0) {
            break;
        }

        const double stage_time_limit =
            std::max(1.0, remaining_time * (stage.time_fraction / total_fraction));
        const auto stage_seed = select_settings(current_settings, stage.indices);
        const size_t stage_dim = stage.indices.size();
        const size_t latent_dim = std::min<size_t>(8, std::max<size_t>(3, stage_dim / 3));
        const auto embedding = make_embedding(stage_dim, latent_dim, rng);

        auto evaluate_latent = [&](const std::vector<double>& latent) {
            const auto stage_settings = decode_latent(stage_seed, embedding, latent, stage.radius);
            const auto merged = merge_settings(current_settings, stage.indices, stage_settings);
            const auto rendered = adaptive_echo::synth(
                merged, time, static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
            ++result.eval_count;
            return std::pair<double, std::vector<float>>(
                static_cast<double>(full_loss(rendered)), stage_settings);
        };

        std::vector<double> current_latent(latent_dim, 0.0);
        auto [current_loss, current_stage_settings] = evaluate_latent(current_latent);
        std::vector<double> best_latent = current_latent;
        auto best_stage_settings = current_stage_settings;
        double best_stage_loss = current_loss;

        std::vector<double> proposal_scales(latent_dim, 0.35);
        double temperature = 0.35;
        int accepted = 0;
        int attempted = 0;

        const auto stage_start = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - stage_start).count() < stage_time_limit) {
            std::vector<std::vector<double>> proposals;
            proposals.reserve(8);
            for (int k = 0; k < 8; ++k) {
                std::vector<double> proposal = current_latent;
                const bool global_jump = (k == 0 && unit_uniform(rng) < 0.2);
                for (size_t d = 0; d < latent_dim; ++d) {
                    const double scale = global_jump ? proposal_scales[d] * 2.5 : proposal_scales[d];
                    proposal[d] = std::clamp(proposal[d] + gaussian(rng) * scale, -1.0, 1.0);
                }
                proposals.push_back(std::move(proposal));
            }

            std::vector<std::vector<float>> settings_batch;
            settings_batch.reserve(proposals.size());
            for (const auto& latent : proposals) {
                const auto stage_settings = decode_latent(stage_seed, embedding, latent, stage.radius);
                settings_batch.push_back(merge_settings(current_settings, stage.indices, stage_settings));
            }
            auto batch_losses = adaptive_echo::compute_audio_loss_batch(settings_batch, full_loss.features());

            for (size_t i = 0; i < proposals.size(); ++i) {
                const double proposal_loss = batch_losses[i];
                const double delta = proposal_loss - current_loss;
                const bool accept =
                    delta <= 0.0 || unit_uniform(rng) < std::exp(-delta / std::max(temperature, 1e-6));
                ++attempted;
                if (accept) {
                    ++accepted;
                    current_latent = proposals[i];
                    current_loss = proposal_loss;
                    current_stage_settings = decode_latent(stage_seed, embedding, current_latent, stage.radius);
                    if (proposal_loss < best_stage_loss) {
                        best_stage_loss = proposal_loss;
                        best_latent = current_latent;
                        best_stage_settings = current_stage_settings;
                    }
                }
            }

            const double progress =
                std::clamp(std::chrono::duration<double>(Clock::now() - stage_start).count() /
                               stage_time_limit,
                           0.0, 1.0);
            temperature = std::max(0.01, 0.35 * std::pow(0.03 / 0.35, progress));

            if (attempted >= 32) {
                const double accept_rate = static_cast<double>(accepted) / attempted;
                const double scale_mult = accept_rate > 0.35 ? 1.12 : (accept_rate < 0.15 ? 0.82 : 0.97);
                for (double& scale : proposal_scales) {
                    scale = std::clamp(scale * scale_mult, 0.01, 0.50);
                }
                accepted = 0;
                attempted = 0;
            }
        }

        current_settings = merge_settings(current_settings, stage.indices, best_stage_settings);
        const auto rendered = adaptive_echo::synth(
            current_settings, time,
            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const float full_stage_loss = full_loss(rendered);
        if (full_stage_loss < result.best_loss) {
            result.best_loss = full_stage_loss;
            result.best_settings = current_settings;
        }
    }

    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - global_start).count();
    if (result.best_loss >= std::numeric_limits<float>::max() * 0.5f) {
        const auto rendered = adaptive_echo::synth(
            current_settings, time,
            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        result.best_loss = full_loss(rendered);
        result.best_settings = current_settings;
    }
    return result;
}

void print_usage() {
    std::cout << "Usage: compare_optimizers <input.wav> [time_limit_seconds] [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --optimizer <all|bayes-tpe|crfmnes>\n";
    std::cout << "  --json\n";
    std::cout << "  --crfmnes-population <int>\n";
    std::cout << "  --crfmnes-sigma <float>\n";
    std::cout << "  --coarse-multiplier <int>\n";
    std::cout << "  --coarse-min-candidates <int>\n";
    std::cout << "  --coarse-wide-noise <float>\n";
    std::cout << "  --coarse-medium-noise <float>\n";
    std::cout << "  --coarse-summary-mix <float>\n";
    std::cout << "  --coarse-uniform-mix <float>\n";
    std::cout << "  --coarse-summary-seed-mix <float>\n";
    std::cout << "  --coarse-default-seed-mix <float>\n";
    std::cout << "  --tpe-gamma <float>\n";
    std::cout << "  --tpe-latent-divisor <int>\n";
    std::cout << "  --tpe-max-latent <int>\n";
    std::cout << "  --tpe-min-latent <int>\n";
    std::cout << "  --tpe-min-init <int>\n";
    std::cout << "  --tpe-init-multiplier <int>\n";
    std::cout << "  --tpe-candidates <int>\n";
    std::cout << "  --tpe-noise-std <float>\n";
    std::cout << "  --tpe-coarse-radius <float>\n";
    std::cout << "  --tpe-shape-radius <float>\n";
    std::cout << "  --tpe-refine-radius <float>\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage();
            return 1;
        }

        const std::string input_path = argv[1];
        int arg_index = 2;
        double time_limit_seconds = 12.0;
        if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
            time_limit_seconds = std::stod(argv[2]);
            arg_index = 3;
        }

        std::string optimizer_mode = "all";
        bool json_output = false;
        int crfmnes_population = adaptive_echo::kDefaultCRFMNESPopulationSize;
        float crfmnes_sigma = adaptive_echo::kDefaultCRFMNESInitialSigma;
        adaptive_echo::CoarseSearchOptions coarse_options;
        TPEHyperparameters tpe_hyperparams;

        auto require_value = [&](int index) {
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value for option: " + std::string(argv[index]));
            }
            return std::string(argv[index + 1]);
        };

        while (arg_index < argc) {
            const std::string option = argv[arg_index];
            if (option == "--optimizer") {
                optimizer_mode = require_value(arg_index);
                arg_index += 2;
            } else if (option == "--json") {
                json_output = true;
                ++arg_index;
            } else if (option == "--crfmnes-population") {
                crfmnes_population = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--crfmnes-sigma") {
                crfmnes_sigma = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--coarse-multiplier") {
                coarse_options.candidate_multiplier = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--coarse-min-candidates") {
                coarse_options.min_candidates = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--coarse-wide-noise") {
                coarse_options.wide_noise_std = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--coarse-medium-noise") {
                coarse_options.medium_noise_std = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--coarse-summary-mix") {
                coarse_options.summary_default_mix = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--coarse-uniform-mix") {
                coarse_options.exploratory_uniform_mix = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--coarse-summary-seed-mix") {
                coarse_options.exploratory_summary_mix = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--coarse-default-seed-mix") {
                coarse_options.exploratory_default_mix = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-gamma") {
                tpe_hyperparams.gamma = std::stod(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-latent-divisor") {
                tpe_hyperparams.latent_divisor = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-max-latent") {
                tpe_hyperparams.max_latent_dim = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-min-latent") {
                tpe_hyperparams.min_latent_dim = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-min-init") {
                tpe_hyperparams.min_init_samples = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-init-multiplier") {
                tpe_hyperparams.init_samples_multiplier = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-candidates") {
                tpe_hyperparams.candidate_count = std::stoi(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-noise-std") {
                tpe_hyperparams.local_noise_std = std::stod(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-coarse-radius") {
                tpe_hyperparams.coarse_radius = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-shape-radius") {
                tpe_hyperparams.shape_radius = std::stof(require_value(arg_index));
                arg_index += 2;
            } else if (option == "--tpe-refine-radius") {
                tpe_hyperparams.refine_radius = std::stof(require_value(arg_index));
                arg_index += 2;
            } else {
                throw std::runtime_error("Unknown option: " + option);
            }
        }

        auto audio = load_audio_file(input_path);
        auto target_audio = adaptive_echo::preprocess_target_audio(
            audio.samples, audio.sample_rate, adaptive_echo::constants::NUM_SAMPLES);

        const auto time = adaptive_echo::make_time_axis(adaptive_echo::constants::NUM_SECONDS,
                                                        adaptive_echo::constants::TRAINING_SAMPLE_RATE);
        adaptive_echo::LossFunction<float> full_loss(target_audio);

        std::optional<adaptive_echo::TrainingResult> crfmnes_result;
        double crfmnes_elapsed = 0.0;
        if (optimizer_mode == "all" || optimizer_mode == "crfmnes") {
            const auto crfmnes_start = std::chrono::steady_clock::now();
            crfmnes_result = adaptive_echo::train_synth_with_coarse_options(
                target_audio, coarse_options, crfmnes_population, crfmnes_sigma,
                static_cast<float>(time_limit_seconds), false);
            crfmnes_elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - crfmnes_start).count();
        }

        std::optional<OptimizationRunResult> bayes_result;
        if (optimizer_mode == "all" || optimizer_mode == "bayes-tpe") {
            bayes_result = run_bayesian_optimizer(target_audio, time_limit_seconds, tpe_hyperparams);
        }

        if (optimizer_mode == "crfmnes") {
            const auto crfmnes_audio =
                adaptive_echo::synth(crfmnes_result->best_settings, time,
                                     static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
            const float crfmnes_loss = full_loss(crfmnes_audio);
            const std::string crfmnes_wav = "build/optimizer_compare/crfmnes_output.wav";
            write_wav_file(crfmnes_wav, crfmnes_audio, adaptive_echo::constants::TRAINING_SAMPLE_RATE);

            if (json_output) {
                std::cout << "{";
                std::cout << "\"optimizer\":\"crfmnes\",";
                std::cout << "\"input\":\"" << input_path << "\",";
                std::cout << "\"loss\":" << std::fixed << std::setprecision(6) << crfmnes_loss << ",";
                std::cout << "\"elapsed\":" << crfmnes_elapsed << ",";
                std::cout << "\"evals\":" << crfmnes_result->final_eval_count << ",";
                std::cout << "\"output_wav\":\"" << crfmnes_wav << "\",";
                std::cout << "\"hyperparameters\":{";
                std::cout << "\"population_size\":" << crfmnes_population << ",";
                std::cout << "\"initial_sigma\":" << crfmnes_sigma << ",";
                std::cout << "\"coarse_candidate_multiplier\":" << coarse_options.candidate_multiplier
                          << ",";
                std::cout << "\"coarse_min_candidates\":" << coarse_options.min_candidates << ",";
                std::cout << "\"coarse_wide_noise_std\":" << coarse_options.wide_noise_std << ",";
                std::cout << "\"coarse_medium_noise_std\":" << coarse_options.medium_noise_std << ",";
                std::cout << "\"coarse_summary_default_mix\":" << coarse_options.summary_default_mix
                          << ",";
                std::cout << "\"coarse_exploratory_uniform_mix\":"
                          << coarse_options.exploratory_uniform_mix << ",";
                std::cout << "\"coarse_exploratory_summary_mix\":"
                          << coarse_options.exploratory_summary_mix << ",";
                std::cout << "\"coarse_exploratory_default_mix\":"
                          << coarse_options.exploratory_default_mix;
                std::cout << "}}";
                std::cout << "\n";
            } else {
                std::cout << "CR-FM-NES | loss " << std::fixed << std::setprecision(4)
                          << crfmnes_loss << " | elapsed " << crfmnes_elapsed << "s | evals "
                          << crfmnes_result->final_eval_count << "\n";
                std::cout << crfmnes_wav << "\n";
            }
            return 0;
        }

        if (optimizer_mode == "bayes-tpe") {
            const auto bayes_audio =
                adaptive_echo::synth(bayes_result->best_settings, time,
                                     static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
            const float bayes_loss = full_loss(bayes_audio);
            const std::string bayes_wav = "build/optimizer_compare/bayes_tpe_output.wav";
            write_wav_file(bayes_wav, bayes_audio, adaptive_echo::constants::TRAINING_SAMPLE_RATE);

            if (json_output) {
                std::cout << "{";
                std::cout << "\"optimizer\":\"bayes-tpe\",";
                std::cout << "\"input\":\"" << input_path << "\",";
                std::cout << "\"loss\":" << std::fixed << std::setprecision(6) << bayes_loss << ",";
                std::cout << "\"elapsed\":" << bayes_result->elapsed_seconds << ",";
                std::cout << "\"evals\":" << bayes_result->eval_count << ",";
                std::cout << "\"output_wav\":\"" << bayes_wav << "\",";
                std::cout << "\"hyperparameters\":{";
                std::cout << "\"gamma\":" << tpe_hyperparams.gamma << ",";
                std::cout << "\"latent_divisor\":" << tpe_hyperparams.latent_divisor << ",";
                std::cout << "\"max_latent_dim\":" << tpe_hyperparams.max_latent_dim << ",";
                std::cout << "\"min_latent_dim\":" << tpe_hyperparams.min_latent_dim << ",";
                std::cout << "\"min_init_samples\":" << tpe_hyperparams.min_init_samples << ",";
                std::cout << "\"init_samples_multiplier\":" << tpe_hyperparams.init_samples_multiplier
                          << ",";
                std::cout << "\"candidate_count\":" << tpe_hyperparams.candidate_count << ",";
                std::cout << "\"local_noise_std\":" << tpe_hyperparams.local_noise_std << ",";
                std::cout << "\"coarse_radius\":" << tpe_hyperparams.coarse_radius << ",";
                std::cout << "\"shape_radius\":" << tpe_hyperparams.shape_radius << ",";
                std::cout << "\"refine_radius\":" << tpe_hyperparams.refine_radius;
                std::cout << "}}";
                std::cout << "\n";
            } else {
                std::cout << "Bayes-TPE | loss " << std::fixed << std::setprecision(4) << bayes_loss
                          << " | elapsed " << bayes_result->elapsed_seconds << "s | evals "
                          << bayes_result->eval_count << "\n";
                std::cout << bayes_wav << "\n";
            }
            return 0;
        }

        auto jade_result = run_jade_optimizer(target_audio, time_limit_seconds);
        auto lshade_result = run_lshade_optimizer(target_audio, time_limit_seconds);
        auto hybrid_result = run_hybrid_optimizer(target_audio, time_limit_seconds);
        auto sa_result = run_simulated_annealing_optimizer(target_audio, time_limit_seconds);

        const auto crfmnes_audio =
            adaptive_echo::synth(crfmnes_result->best_settings, time,
                                 static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const auto bayes_audio =
            adaptive_echo::synth(bayes_result->best_settings, time,
                                 static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const auto jade_audio =
            adaptive_echo::synth(jade_result.best_settings, time,
                                 static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const auto lshade_audio =
            adaptive_echo::synth(lshade_result.best_settings, time,
                                 static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const auto hybrid_audio =
            adaptive_echo::synth(hybrid_result.best_settings, time,
                                 static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));
        const auto sa_audio =
            adaptive_echo::synth(sa_result.best_settings, time,
                                 static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE));

        const float crfmnes_loss = full_loss(crfmnes_audio);
        const float bayes_loss = full_loss(bayes_audio);
        const float jade_loss = full_loss(jade_audio);
        const float lshade_loss = full_loss(lshade_audio);
        const float hybrid_loss = full_loss(hybrid_audio);
        const float sa_loss = full_loss(sa_audio);

        const std::string crfmnes_wav = "build/optimizer_compare/crfmnes_output.wav";
        const std::string bayes_wav = "build/optimizer_compare/bayes_tpe_output.wav";
        const std::string jade_wav = "build/optimizer_compare/jade_output.wav";
        const std::string lshade_wav = "build/optimizer_compare/lshade_output.wav";
        const std::string hybrid_wav = "build/optimizer_compare/tpe_jade_hybrid_output.wav";
        const std::string sa_wav = "build/optimizer_compare/simulated_annealing_output.wav";
        write_wav_file(crfmnes_wav, crfmnes_audio, adaptive_echo::constants::TRAINING_SAMPLE_RATE);
        write_wav_file(bayes_wav, bayes_audio, adaptive_echo::constants::TRAINING_SAMPLE_RATE);
        write_wav_file(jade_wav, jade_audio, adaptive_echo::constants::TRAINING_SAMPLE_RATE);
        write_wav_file(lshade_wav, lshade_audio, adaptive_echo::constants::TRAINING_SAMPLE_RATE);
        write_wav_file(hybrid_wav, hybrid_audio, adaptive_echo::constants::TRAINING_SAMPLE_RATE);
        write_wav_file(sa_wav, sa_audio, adaptive_echo::constants::TRAINING_SAMPLE_RATE);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Input: " << input_path << "\n";
        std::cout << "Time limit per optimizer: " << time_limit_seconds << "s\n\n";
        std::cout << "Optimizer comparison\n";
        std::cout << "-----------------------------------------------\n";
        std::cout << "CR-FM-NES  | loss " << crfmnes_loss << " | elapsed " << crfmnes_elapsed
                  << "s | evals " << crfmnes_result->final_eval_count << "\n";
        std::cout << "Bayes-TPE  | loss " << bayes_loss << " | elapsed "
                  << bayes_result->elapsed_seconds << "s | evals " << bayes_result->eval_count
                  << "\n";
        std::cout << "JADE-DE    | loss " << jade_loss << " | elapsed "
                  << jade_result.elapsed_seconds << "s | evals " << jade_result.eval_count
                  << "\n";
        std::cout << "L-SHADE    | loss " << lshade_loss << " | elapsed "
                  << lshade_result.elapsed_seconds << "s | evals " << lshade_result.eval_count
                  << "\n";
        std::cout << "TPE+JADE   | loss " << hybrid_loss << " | elapsed "
                  << hybrid_result.elapsed_seconds << "s | evals " << hybrid_result.eval_count
                  << "\n";
        std::cout << "Fast-SA    | loss " << sa_loss << " | elapsed "
                  << sa_result.elapsed_seconds << "s | evals " << sa_result.eval_count
                  << "\n\n";
        std::cout << "Rendered outputs\n";
        std::cout << "-----------------------------------------------\n";
        std::cout << crfmnes_wav << "\n";
        std::cout << bayes_wav << "\n";
        std::cout << jade_wav << "\n";
        std::cout << lshade_wav << "\n";
        std::cout << hybrid_wav << "\n";
        std::cout << sa_wav << "\n";

        const float best_loss =
            std::min({crfmnes_loss, bayes_loss, jade_loss, lshade_loss, hybrid_loss, sa_loss});
        if (best_loss == sa_loss && sa_loss < crfmnes_loss && sa_loss < bayes_loss &&
            sa_loss < jade_loss && sa_loss < lshade_loss && sa_loss < hybrid_loss) {
            std::cout << "\nWinner: Fast-SA\n";
        } else if (best_loss == hybrid_loss && hybrid_loss < crfmnes_loss && hybrid_loss < bayes_loss &&
                   hybrid_loss < jade_loss && hybrid_loss < lshade_loss && hybrid_loss < sa_loss) {
            std::cout << "\nWinner: TPE+JADE\n";
        } else if (best_loss == lshade_loss && lshade_loss < crfmnes_loss &&
                   lshade_loss < bayes_loss && lshade_loss < jade_loss &&
                   lshade_loss < hybrid_loss && lshade_loss < sa_loss) {
            std::cout << "\nWinner: L-SHADE\n";
        } else if (best_loss == jade_loss && jade_loss < crfmnes_loss && jade_loss < bayes_loss &&
                   jade_loss < lshade_loss && jade_loss < hybrid_loss && jade_loss < sa_loss) {
            std::cout << "\nWinner: JADE-DE\n";
        } else if (best_loss == bayes_loss && bayes_loss < crfmnes_loss &&
                   bayes_loss < jade_loss && bayes_loss < lshade_loss &&
                   bayes_loss < hybrid_loss && bayes_loss < sa_loss) {
            std::cout << "\nWinner: Bayes-TPE\n";
        } else if (best_loss == crfmnes_loss && crfmnes_loss < bayes_loss &&
                   crfmnes_loss < jade_loss && crfmnes_loss < lshade_loss &&
                   crfmnes_loss < hybrid_loss && crfmnes_loss < sa_loss) {
            std::cout << "\nWinner: CR-FM-NES\n";
        } else {
            std::cout << "\nWinner: tie\n";
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "compare_optimizers failed: " << error.what() << "\n";
        return 1;
    }
}
