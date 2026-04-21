#include "adaptive_echo/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/interpolation.hpp"
#include "adaptive_echo/loss.hpp"
#include "adaptive_echo/resample.hpp"
#include "adaptive_echo/synth.hpp"

namespace adaptive_echo {
namespace {

constexpr float kMinFrequencyHz = 50.0f;
constexpr float kMaxFrequencyHz = 2000.0f;
constexpr float kDefaultReferenceFrequencyHz = 440.0f;
constexpr float kMinEnvelopeLengthSeconds = 0.2f;
constexpr float kMaxEnvelopeLengthSeconds =
    static_cast<float>(constants::NUM_SECONDS);
constexpr float kMinEnvelopeAttackSeconds = 0.05f;
constexpr float kMaxEnvelopeAttackSeconds = 0.5f;
constexpr float kMinEnvelopeDecaySeconds = 0.05f;
constexpr float kMaxEnvelopeDecaySeconds = 0.5f;
constexpr float kMinEnvelopeReleaseSeconds = 0.05f;
constexpr float kMaxEnvelopeReleaseSeconds = 0.5f;
constexpr float kMinFilterCutoffHz = 20.0f;
constexpr float kMaxFilterCutoffHz = 20000.0f;
constexpr int kRefinementPopulationSize = 16;

struct TargetAudioSummary {
    float dominant_frequency_hz = 440.0f;
    float spectral_centroid_hz = 1000.0f;
    float spectral_rolloff_hz = 9000.0f;
    float spectral_flatness = 0.1f;
    float rms = 0.15f;
    float crest_factor = 4.0f;
    float active_duration_seconds = 0.6f;
    float attack_seconds = 0.05f;
    float decay_seconds = 0.18f;
    float sustain_level = 0.4f;
    float release_seconds = 0.12f;
};

struct CandidateResult {
    std::vector<float> settings;
    float full_loss = std::numeric_limits<float>::max();
    int eval_count = 0;
    int iterations_completed = 0;
};

struct TPEHyperparameters {
    float gamma = 0.2633971f;
    int latent_divisor = 4;
    int max_latent_dim = 9;
    int min_latent_dim = 4;
    int min_init_samples = 8;
    int init_samples_multiplier = 3;
    int candidate_count = 630;
    float local_noise_std = 0.0664878f;
    float coarse_radius = 0.2591669f;
    float shape_radius = 0.1033653f;
    float refine_radius = 0.1317812f;
};

struct TrainingStageConfig {
    const char* name = "";
    std::vector<int> indices;
    float time_fraction = 1.0f;
    float radius = 0.25f;
};

class TPEModel {
   public:
    explicit TPEModel(float gamma) : gamma_(gamma) {}

    bool fit(const std::vector<std::vector<float>>& inputs, const std::vector<float>& values) {
        if (inputs.empty() || inputs.size() != values.size()) {
            return false;
        }

        const size_t dim = inputs.front().size();
        const size_t n = values.size();
        const size_t good_count =
            std::clamp(static_cast<size_t>(std::ceil(gamma_ * static_cast<float>(n))),
                       static_cast<size_t>(1), n);

        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t lhs, size_t rhs) { return values[lhs] < values[rhs]; });

        good_means_.assign(dim, 0.0f);
        good_stds_.assign(dim, 0.28f);
        bad_means_.assign(dim, 0.0f);
        bad_stds_.assign(dim, 0.55f);

        for (size_t d = 0; d < dim; ++d) {
            float good_mean = 0.0f;
            float bad_mean = 0.0f;
            for (size_t i = 0; i < good_count; ++i) {
                good_mean += inputs[order[i]][d];
            }
            for (size_t i = good_count; i < n; ++i) {
                bad_mean += inputs[order[i]][d];
            }
            good_mean /= static_cast<float>(good_count);
            bad_mean /= static_cast<float>(std::max<size_t>(1, n - good_count));

            float good_var = 0.0f;
            float bad_var = 0.0f;
            for (size_t i = 0; i < good_count; ++i) {
                const float diff = inputs[order[i]][d] - good_mean;
                good_var += diff * diff;
            }
            for (size_t i = good_count; i < n; ++i) {
                const float diff = inputs[order[i]][d] - bad_mean;
                bad_var += diff * diff;
            }

            good_means_[d] = good_mean;
            bad_means_[d] = bad_mean;
            good_stds_[d] =
                std::clamp(std::sqrt(good_var / static_cast<float>(good_count) + 1e-4f), 0.06f,
                           0.45f);
            bad_stds_[d] = std::clamp(
                std::sqrt(bad_var / static_cast<float>(std::max<size_t>(1, n - good_count)) +
                          1e-4f),
                0.10f, 0.75f);
        }

        best_point_ = inputs[order.front()];
        return true;
    }

    std::vector<float> sample_candidate(std::mt19937& rng) const {
        std::vector<float> candidate(best_point_.size(), 0.0f);
        for (size_t d = 0; d < candidate.size(); ++d) {
            std::normal_distribution<float> distribution(good_means_[d], good_stds_[d]);
            candidate[d] = std::clamp(distribution(rng), -1.0f, 1.0f);
        }
        return candidate;
    }

    float score_candidate(const std::vector<float>& candidate) const {
        float good_log_density = 0.0f;
        float bad_log_density = 0.0f;
        for (size_t d = 0; d < candidate.size(); ++d) {
            good_log_density += gaussian_log_pdf(candidate[d], good_means_[d], good_stds_[d]);
            bad_log_density += gaussian_log_pdf(candidate[d], bad_means_[d], bad_stds_[d]);
        }
        return good_log_density - bad_log_density;
    }

   private:
    static float gaussian_log_pdf(float x, float mean, float stddev) {
        const float safe_stddev = std::max(stddev, 1e-4f);
        const float z = (x - mean) / safe_stddev;
        return -std::log(safe_stddev) - 0.5f * z * z -
               0.5f * std::log(2.0f * static_cast<float>(M_PI));
    }

    float gamma_ = 0.20f;
    std::vector<float> good_means_;
    std::vector<float> good_stds_;
    std::vector<float> bad_means_;
    std::vector<float> bad_stds_;
    std::vector<float> best_point_;
};

float sanitize_frequency(float frequency_hz) {
    return std::clamp(frequency_hz, kMinFrequencyHz, kMaxFrequencyHz);
}

float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float inverse_linear(float minimum, float maximum, float value) {
    const float span = std::max(maximum - minimum, 1e-6f);
    return clamp01((value - minimum) / span);
}

float inverse_exp_interp(float minimum, float maximum, float value) {
    value = std::clamp(value, minimum, maximum);
    const float safe_minimum = std::max(minimum, 1e-6f);
    const float safe_maximum = std::max(maximum, safe_minimum + 1e-6f);
    return clamp01(std::log(value / safe_minimum) / std::log(safe_maximum / safe_minimum));
}

float hz_to_filter_normalized(float cutoff_hz) {
    cutoff_hz = std::clamp(cutoff_hz, kMinFilterCutoffHz, kMaxFilterCutoffHz);
    return clamp01(std::log(cutoff_hz / kMinFilterCutoffHz) / std::log(1000.0f));
}

float elapsed_seconds_since(const std::chrono::steady_clock::time_point& start_time);

uint32_t make_loss_seed(uint32_t requested_seed) {
    if (requested_seed != 0) {
        return requested_seed;
    }

    std::random_device device;
    const uint32_t seed = device();
    if (seed != 0) {
        return seed;
    }

    return static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

void normalize_audio(std::vector<float>& audio, float target_peak) {
    float max_value = 0.0f;
    for (float sample : audio) {
        max_value = std::max(max_value, std::abs(sample));
    }

    if (max_value <= 0.0f) {
        return;
    }

    const float scale = target_peak / max_value;
    for (float& sample : audio) {
        sample *= scale;
    }
}

float evaluate_settings(const std::vector<float>& settings,
                        const std::vector<float>& time,
                        const LossFunction<float>& loss_fn) {
    return loss_fn(synth(settings, time, static_cast<float>(constants::TRAINING_SAMPLE_RATE)));
}

std::vector<float> evaluate_settings_batch(const std::vector<std::vector<float>>& settings_batch,
                                           const std::vector<float>& time,
                                           const LossFunction<float>& loss_fn) {
    return loss_fn.evaluate_generated_batch(settings_batch.size(), [&](size_t i) {
        return synth(settings_batch[i], time, static_cast<float>(constants::TRAINING_SAMPLE_RATE));
    });
}

CandidateResult run_coarse_search(const std::vector<float>& summary_seed,
                                  const std::vector<float>& time,
                                  const LossFunction<float>& loss_fn,
                                  int population_size,
                                  const CoarseSearchOptions& coarse_options,
                                  float time_limit_seconds, float coarse_sigma,
                                  const TrainingProgressCallback& progress_callback) {
    auto& rng = detail::get_crfmnes_rng();
    std::normal_distribution<float> wide_noise(0.0f, coarse_options.wide_noise_std);
    std::normal_distribution<float> medium_noise(0.0f, coarse_options.medium_noise_std);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    const auto default_seed = default_settings();
    const size_t num_settings = summary_seed.size();
    const int candidate_count =
        std::max(coarse_options.min_candidates,
                 population_size * std::max(1, coarse_options.candidate_multiplier));

    std::vector<float> midpoint(num_settings, 0.5f);
    for (size_t i = 0; i < num_settings; ++i) {
        midpoint[i] = clamp01(0.5f * (summary_seed[i] + default_seed[i]));
    }
    CandidateResult result;
    result.settings = summary_seed;
    result.full_loss = evaluate_settings(summary_seed, time, loss_fn);
    result.eval_count = 1;

    const auto coarse_start = std::chrono::steady_clock::now();
    bool first_batch = true;
    while (first_batch ||
           std::chrono::duration_cast<std::chrono::duration<float>>(
               std::chrono::steady_clock::now() - coarse_start)
                   .count() < time_limit_seconds) {
        std::vector<std::vector<float>> candidates;
        candidates.reserve(static_cast<size_t>(candidate_count));
        candidates.push_back(result.settings);
        candidates.push_back(summary_seed);
        candidates.push_back(default_seed);
        candidates.push_back(midpoint);

        while (static_cast<int>(candidates.size()) < candidate_count) {
            const size_t mode = candidates.size() % 4;
            std::vector<float> candidate(num_settings, 0.5f);

            for (size_t i = 0; i < num_settings; ++i) {
                const float base = (mode == 0) ? result.settings[i]
                                 : (mode == 1) ? summary_seed[i]
                                 : (mode == 2) ? midpoint[i]
                                               : uniform01(rng);
                float value = base;
                if (mode == 0) {
                    value += medium_noise(rng);
                } else if (mode == 1) {
                    value += wide_noise(rng);
                } else if (mode == 2) {
                    value = coarse_options.summary_default_mix * summary_seed[i] +
                            (1.0f - coarse_options.summary_default_mix) * default_seed[i] +
                            wide_noise(rng);
                } else {
                    value = coarse_options.exploratory_uniform_mix * uniform01(rng) +
                            coarse_options.exploratory_summary_mix * summary_seed[i] +
                            coarse_options.exploratory_default_mix * default_seed[i];
                }
                candidate[i] = clamp01(value);
            }

            candidates.push_back(std::move(candidate));
        }

        const auto losses = evaluate_settings_batch(candidates, time, loss_fn);
        result.eval_count += static_cast<int>(losses.size());
        const auto best_it = std::min_element(losses.begin(), losses.end());
        if (best_it != losses.end()) {
            const size_t best_index = static_cast<size_t>(std::distance(losses.begin(), best_it));
            if (losses[best_index] < result.full_loss) {
                result.settings = candidates[best_index];
                result.full_loss = losses[best_index];
            }
        }
        ++result.iterations_completed;
        if (progress_callback) {
            progress_callback(TrainingProgress {result.iterations_completed, result.full_loss,
                                                coarse_sigma, result.eval_count,
                                                elapsed_seconds_since(coarse_start)});
        }
        first_batch = false;
    }

    return result;
}

TargetAudioSummary summarize_target_audio(
    const std::vector<float>& target_audio,
    const TargetFeaturesFast<float>& features) {
    TargetAudioSummary summary;

    if (!features.stfts.empty() && !features.stfts.front().empty() &&
        !features.num_freqs.empty() && !features.fft_sizes.empty()) {
        const size_t scale = 0;
        const size_t num_freqs = features.num_freqs[scale];
        const size_t num_frames = features.num_frames[scale];
        const float sample_rate = static_cast<float>(constants::TRAINING_SAMPLE_RATE);
        const float n_fft = static_cast<float>(features.fft_sizes[scale]);
        std::vector<float> mean_spectrum(num_freqs, 0.0f);
        for (size_t frame = 0; frame < num_frames; ++frame) {
            const size_t offset = frame * num_freqs;
            for (size_t bin = 0; bin < num_freqs; ++bin) {
                mean_spectrum[bin] += features.stfts[scale][offset + bin];
            }
        }
        if (num_frames > 0) {
            const float inv_frames = 1.0f / static_cast<float>(num_frames);
            for (float& value : mean_spectrum) {
                value *= inv_frames;
            }
        }

        float weighted_sum = 0.0f;
        float magnitude_sum = 0.0f;
        float log_sum = 0.0f;
        float max_magnitude = 0.0f;
        size_t max_index = 0;
        for (size_t bin = 1; bin < num_freqs; ++bin) {
            const float frequency = static_cast<float>(bin) * sample_rate / n_fft;
            const float magnitude = std::max(mean_spectrum[bin], 1e-8f);
            weighted_sum += frequency * magnitude;
            magnitude_sum += magnitude;
            log_sum += std::log(magnitude);
            if (frequency >= kMinFrequencyHz && frequency <= kMaxFrequencyHz &&
                magnitude > max_magnitude) {
                max_magnitude = magnitude;
                max_index = bin;
            }
        }

        if (magnitude_sum > 0.0f) {
            summary.spectral_centroid_hz = weighted_sum / magnitude_sum;
            const float geometric_mean =
                std::exp(log_sum / std::max(1.0f, static_cast<float>(num_freqs - 1)));
            summary.spectral_flatness = clamp01(
                geometric_mean * static_cast<float>(num_freqs - 1) / magnitude_sum);

            float cumulative = 0.0f;
            for (size_t bin = 1; bin < num_freqs; ++bin) {
                cumulative += mean_spectrum[bin];
                if (cumulative >= 0.90f * magnitude_sum) {
                    summary.spectral_rolloff_hz =
                        static_cast<float>(bin) * sample_rate / n_fft;
                    break;
                }
            }
        }

        if (max_index > 0) {
            summary.dominant_frequency_hz =
                static_cast<float>(max_index) * sample_rate / n_fft;
        }
    }

    if (!target_audio.empty()) {
        const size_t window = 256;
        const size_t num_blocks = (target_audio.size() + window - 1) / window;
        std::vector<float> envelope(num_blocks, 0.0f);
        float peak = 0.0f;
        double sum_squares = 0.0;
        for (size_t i = 0; i < target_audio.size(); ++i) {
            const float sample = std::abs(target_audio[i]);
            envelope[i / window] += sample;
            peak = std::max(peak, sample);
            sum_squares += static_cast<double>(target_audio[i]) * target_audio[i];
        }
        for (float& value : envelope) {
            value /= static_cast<float>(window);
        }

        summary.rms = std::sqrt(static_cast<float>(
            sum_squares / std::max<size_t>(1, target_audio.size())));
        summary.crest_factor = peak / std::max(summary.rms, 1e-6f);

        const auto peak_it = std::max_element(envelope.begin(), envelope.end());
        const float envelope_peak = peak_it != envelope.end() ? *peak_it : 0.0f;
        const float threshold = std::max(1e-4f, envelope_peak * 0.1f);
        size_t first_active = 0;
        size_t last_active = envelope.empty() ? 0 : envelope.size() - 1;
        while (first_active < envelope.size() && envelope[first_active] < threshold) {
            ++first_active;
        }
        while (last_active > first_active && envelope[last_active] < threshold) {
            --last_active;
        }

        if (first_active < envelope.size() && last_active >= first_active) {
            const float seconds_per_block =
                static_cast<float>(window) / static_cast<float>(constants::TRAINING_SAMPLE_RATE);
            const size_t peak_index = peak_it != envelope.end()
                                          ? static_cast<size_t>(std::distance(envelope.begin(), peak_it))
                                          : first_active;
            summary.active_duration_seconds = std::clamp(
                static_cast<float>(last_active - first_active + 1) * seconds_per_block,
                kMinEnvelopeLengthSeconds, kMaxEnvelopeLengthSeconds);
            summary.attack_seconds = std::clamp(
                static_cast<float>(std::max<size_t>(
                    1, peak_index > first_active ? peak_index - first_active : 1)) *
                    seconds_per_block,
                kMinEnvelopeAttackSeconds, kMaxEnvelopeAttackSeconds);
            summary.decay_seconds = std::clamp(summary.active_duration_seconds * 0.25f,
                                               kMinEnvelopeDecaySeconds,
                                               kMaxEnvelopeDecaySeconds);
            summary.release_seconds = std::clamp(summary.active_duration_seconds * 0.18f,
                                                 kMinEnvelopeReleaseSeconds,
                                                 kMaxEnvelopeReleaseSeconds);

            const size_t tail_span =
                std::max<size_t>(1, (last_active - first_active + 1) / 5);
            float tail_mean = 0.0f;
            for (size_t i = last_active + 1 - tail_span; i <= last_active; ++i) {
                tail_mean += envelope[i];
            }
            tail_mean /= static_cast<float>(tail_span);
            summary.sustain_level = clamp01(
                envelope_peak > 0.0f ? tail_mean / envelope_peak : 0.0f);
        }
    }

    summary.dominant_frequency_hz = sanitize_frequency(summary.dominant_frequency_hz);
    summary.spectral_centroid_hz =
        std::clamp(summary.spectral_centroid_hz, kMinFrequencyHz, kMaxFilterCutoffHz);
    summary.spectral_rolloff_hz =
        std::clamp(summary.spectral_rolloff_hz, kMinFrequencyHz, kMaxFilterCutoffHz);
    return summary;
}

std::vector<float> make_seed_from_summary(const TargetAudioSummary& summary) {
    auto seed = default_settings();

    const float frequency_norm = hz_to_normalized_frequency(summary.dominant_frequency_hz);
    const float brightness = clamp01(
        summary.spectral_centroid_hz /
        std::max(summary.dominant_frequency_hz * 4.0f, 1.0f));
    const float noisiness = clamp01(summary.spectral_flatness * 1.8f);
    const float modulation_ratio = clamp01(
        summary.spectral_centroid_hz /
            std::max(summary.dominant_frequency_hz * 2.0f, 1.0f) -
        0.35f);

    const float vol_length = inverse_exp_interp(
        kMinEnvelopeLengthSeconds, kMaxEnvelopeLengthSeconds,
        summary.active_duration_seconds);
    const float attack = inverse_exp_interp(
        kMinEnvelopeAttackSeconds, kMaxEnvelopeAttackSeconds,
        summary.attack_seconds);
    const float decay = inverse_exp_interp(
        kMinEnvelopeDecaySeconds, kMaxEnvelopeDecaySeconds,
        summary.decay_seconds);
    const float sustain = inverse_linear(0.1f, 1.0f, summary.sustain_level);
    const float release = inverse_exp_interp(
        kMinEnvelopeReleaseSeconds, kMaxEnvelopeReleaseSeconds,
        summary.release_seconds);

    for (int offset : {0, 5}) {
        seed[static_cast<size_t>(offset + 0)] = vol_length;
        seed[static_cast<size_t>(offset + 1)] = attack;
        seed[static_cast<size_t>(offset + 2)] = decay;
        seed[static_cast<size_t>(offset + 3)] = sustain;
        seed[static_cast<size_t>(offset + 4)] = release;
    }

    seed[10] = clamp01(vol_length * 0.9f);
    seed[11] = clamp01(attack * 0.8f);
    seed[12] = clamp01(decay);
    seed[13] = clamp01(0.35f + brightness * 0.45f);
    seed[14] = clamp01(release * 0.7f);

    seed[39] = clamp01(vol_length * 0.85f);
    seed[40] = clamp01(attack * 0.6f);
    seed[41] = clamp01(decay * 0.8f);
    seed[42] = clamp01(0.25f + modulation_ratio * 0.5f);
    seed[43] = clamp01(release * 0.8f);

    seed[constants::OSC_A_FREQ_LOW_INDEX] = frequency_norm;
    seed[constants::OSC_A_FREQ_HIGH_INDEX] = frequency_norm;
    seed[17] = 0.5f;
    seed[18] = 0.5f;
    seed[19] = clamp01(0.65f - brightness * 0.25f);
    seed[20] = clamp01(seed[19] * 0.9f);
    seed[21] = clamp01(0.25f + brightness * 0.45f);
    seed[22] = clamp01(seed[21] * 0.9f);
    seed[23] = clamp01(0.65f + summary.rms * 0.7f);
    seed[24] = seed[23];
    seed[25] = clamp01(noisiness * 0.6f);
    seed[26] = seed[25];

    const float modulator_hz = sanitize_frequency(
        summary.dominant_frequency_hz *
        std::clamp(1.0f + brightness * 1.5f + modulation_ratio, 1.0f, 4.0f));
    const float modulator_norm = hz_to_normalized_frequency(modulator_hz);
    seed[constants::OSC_B_FREQ_LOW_INDEX] = modulator_norm;
    seed[constants::OSC_B_FREQ_HIGH_INDEX] = modulator_norm;
    seed[29] = 0.5f;
    seed[30] = 0.5f;
    seed[31] = clamp01(0.45f - brightness * 0.15f);
    seed[32] = seed[31];
    seed[33] = clamp01(0.35f + brightness * 0.55f);
    seed[34] = seed[33];
    seed[35] = clamp01(0.25f + modulation_ratio * 0.45f);
    seed[36] = seed[35];
    seed[37] = clamp01(noisiness * 0.8f);
    seed[38] = seed[37];

    const float fm_amount = clamp01(modulation_ratio * 0.75f);
    seed[44] = fm_amount;
    seed[45] = clamp01(fm_amount + 0.1f);

    seed[constants::HIGH_PASS_CUTOFF_INDEX] = hz_to_filter_normalized(
        std::max(kMinFilterCutoffHz, summary.dominant_frequency_hz * 0.35f));
    seed[constants::HIGH_PASS_SLOPE_INDEX] = clamp01(noisiness * 0.35f);
    seed[constants::LOW_PASS_CUTOFF_INDEX] = hz_to_filter_normalized(
        std::min(kMaxFilterCutoffHz,
                 summary.spectral_rolloff_hz * (1.0f + brightness * 0.35f)));
    seed[constants::LOW_PASS_SLOPE_INDEX] = clamp01(0.2f + brightness * 0.45f);
    seed[constants::DISTORTION_INDEX] =
        clamp01((2.8f - summary.crest_factor) * 0.22f + brightness * 0.2f);

    return seed;
}

std::vector<float> jitter_settings(const std::vector<float>& base, float jitter,
                                   uint32_t seed_value) {
    auto result = base;
    std::mt19937 rng(seed_value);
    std::normal_distribution<float> noise(0.0f, jitter);
    for (float& value : result) {
        value = clamp01(value + noise(rng));
    }
    return result;
}

float elapsed_seconds_since(
    const std::chrono::steady_clock::time_point& start_time) {
    return std::chrono::duration_cast<std::chrono::duration<float>>(
               std::chrono::steady_clock::now() - start_time)
        .count();
}

float remaining_seconds(const std::chrono::steady_clock::time_point& start_time,
                        float time_limit) {
    if (time_limit <= 0.0f) {
        return std::numeric_limits<float>::max();
    }
    return std::max(0.0f, time_limit - elapsed_seconds_since(start_time));
}

void report_progress(const TrainingProgressCallback& progress_callback,
                     int generation_offset, int eval_offset, float elapsed_offset,
                     float time_limit, float best_loss,
                     const TrainingProgress& progress) {
    if (!progress_callback) {
        return;
    }
    progress_callback(TrainingProgress {
        generation_offset + progress.generation,
        std::isfinite(best_loss) ? best_loss : progress.best_loss,
        progress.sigma,
        eval_offset + progress.eval_count,
        time_limit > 0.0f
            ? std::min(time_limit, elapsed_offset + progress.elapsed_seconds)
            : elapsed_offset + progress.elapsed_seconds,
    });
}

std::vector<float> decode_latent(const std::vector<float>& seed,
                                 const std::vector<std::vector<float>>& embedding,
                                 const std::vector<float>& latent, float radius) {
    std::vector<float> decoded(seed);
    for (size_t row = 0; row < decoded.size(); ++row) {
        float delta = 0.0f;
        for (size_t col = 0; col < latent.size(); ++col) {
            delta += embedding[row][col] * latent[col];
        }
        decoded[row] = clamp01(seed[row] + radius * delta);
    }
    return decoded;
}

std::vector<std::vector<float>> make_embedding(size_t rows, size_t cols, std::mt19937& rng) {
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    std::vector<std::vector<float>> embedding(rows, std::vector<float>(cols, 0.0f));
    for (size_t row = 0; row < rows; ++row) {
        float norm = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float value = gaussian(rng);
            embedding[row][col] = value;
            norm += value * value;
        }
        norm = std::sqrt(std::max(norm, 1e-9f));
        for (size_t col = 0; col < cols; ++col) {
            embedding[row][col] /= norm;
        }
    }
    return embedding;
}

std::vector<int> all_indices() {
    std::vector<int> indices(constants::NUM_SETTINGS);
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

CRFMNESResult<float> coordinate_refine(const std::vector<float>& initial_settings,
                                       const std::vector<float>& time,
                                       const LossFunction<float>& loss_fn,
                                       float time_limit, bool verbose,
                                       const TrainingProgressCallback& progress_callback,
                                       int generation_offset, int eval_offset,
                                       float elapsed_offset, float best_loss_so_far) {
    CRFMNESResult<float> result;
    result.best_settings = initial_settings;
    result.best_loss = evaluate_settings(initial_settings, time, loss_fn);
    result.final_eval_count = 1;
    result.final_sigma = 0.08f;

    auto start_time = std::chrono::steady_clock::now();
    std::vector<float> steps(initial_settings.size(), 0.08f);

    while (remaining_seconds(start_time, time_limit) > 0.0f) {
        bool improved = false;
        for (size_t index = 0; index < result.best_settings.size(); ++index) {
            if (remaining_seconds(start_time, time_limit) <= 0.0f) {
                break;
            }

            const float step = steps[index];
            if (step < 0.003f) {
                continue;
            }

            const float base = result.best_settings[index];
            float best_candidate_loss = result.best_loss;
            std::vector<float> best_candidate = result.best_settings;

            for (float direction : {-1.0f, 1.0f}) {
                auto candidate = result.best_settings;
                candidate[index] = clamp01(base + direction * step);
                if (candidate[index] == base) {
                    continue;
                }
                const float candidate_loss =
                    evaluate_settings(candidate, time, loss_fn);
                ++result.final_eval_count;
                if (candidate_loss < best_candidate_loss) {
                    best_candidate_loss = candidate_loss;
                    best_candidate = std::move(candidate);
                }
            }

            if (best_candidate_loss < result.best_loss) {
                result.best_loss = best_candidate_loss;
                result.best_settings = std::move(best_candidate);
                best_loss_so_far = std::min(best_loss_so_far, result.best_loss);
                improved = true;
            } else {
                steps[index] *= 0.75f;
            }
        }

        ++result.iterations_completed;
        result.final_sigma = *std::max_element(steps.begin(), steps.end());
        report_progress(progress_callback, generation_offset, eval_offset, elapsed_offset,
                        time_limit, best_loss_so_far,
                        TrainingProgress {result.iterations_completed, result.best_loss,
                                          result.final_sigma, result.final_eval_count,
                                          elapsed_seconds_since(start_time)});

        if (!improved && result.final_sigma < 0.003f) {
            break;
        }
        if (!improved) {
            for (float& step : steps) {
                step *= 0.5f;
            }
        }
    }

    if (verbose) {
        std::cout << "Coordinate refine: Best Loss = " << result.best_loss
                  << " | Evals = " << result.final_eval_count << std::endl;
    }

    return result;
}

}  // namespace

std::vector<float> default_settings() {
    return std::vector<float>(constants::NUM_SETTINGS, 0.5f);
}

std::vector<float> make_time_axis(double duration_seconds, double sample_rate) {
    const auto sample_count =
        std::max(1, static_cast<int>(std::ceil(duration_seconds * sample_rate)));
    std::vector<float> time(static_cast<size_t>(sample_count));

    if (sample_count == 1) {
        time[0] = 0.0f;
        return time;
    }

    const auto step = static_cast<float>(duration_seconds / static_cast<double>(sample_count - 1));
    for (int i = 0; i < sample_count; ++i) {
        time[static_cast<size_t>(i)] = static_cast<float>(i) * step;
    }
    return time;
}

std::vector<float> preprocess_target_audio(const std::vector<float>& audio, double input_sample_rate,
                                           int target_length) {
    std::vector<float> processed = audio;
    if (processed.empty() || input_sample_rate <= 0.0) {
        return std::vector<float>(static_cast<size_t>(target_length), 0.0f);
    }

    if (std::abs(input_sample_rate - constants::TRAINING_SAMPLE_RATE) > 0.5) {
        const auto ratio = static_cast<float>(constants::TRAINING_SAMPLE_RATE / input_sample_rate);
        const auto new_length = std::max(1, static_cast<int>(processed.size() * ratio));
        processed = resample_fft(processed, static_cast<size_t>(new_length));
    }

    if (static_cast<int>(processed.size()) > target_length) {
        processed.resize(static_cast<size_t>(target_length));
    } else {
        processed.resize(static_cast<size_t>(target_length), 0.0f);
    }

    normalize_audio(processed, 1.0f);
    normalize_audio(processed, 0.5f);
    return processed;
}

float map_normalized_envelope_length(float value) {
    return exp_interp(0.2f, static_cast<float>(constants::NUM_SECONDS), value);
}

double max_envelope_duration_seconds(const std::vector<float>& settings) {
    if (settings.size() < constants::NUM_SETTINGS) {
        return static_cast<double>(constants::NUM_SECONDS);
    }

    const std::array<int, 4> envelope_length_indices = {0, 5, 10, 39};
    double max_duration = 0.2;
    for (int index : envelope_length_indices) {
        max_duration = std::max(max_duration, static_cast<double>(map_normalized_envelope_length(
                                                   settings[static_cast<size_t>(index)])));
    }
    return max_duration;
}

float normalized_frequency_to_hz(float normalized_value) {
    const float min_frequency_log = 12.0f * std::log2(kMinFrequencyHz);
    const float max_frequency_log = 12.0f * std::log2(kMaxFrequencyHz);
    const float semitones =
        min_frequency_log + (max_frequency_log - min_frequency_log) * normalized_value;
    return std::pow(2.0f, semitones / 12.0f);
}

float hz_to_normalized_frequency(float hz) {
    const float clamped_hz = sanitize_frequency(hz);
    const float min_frequency_log = 12.0f * std::log2(kMinFrequencyHz);
    const float max_frequency_log = 12.0f * std::log2(kMaxFrequencyHz);
    const float semitones = 12.0f * std::log2(clamped_hz);
    return std::clamp((semitones - min_frequency_log) / (max_frequency_log - min_frequency_log),
                      0.0f, 1.0f);
}

std::vector<float> retune_settings_for_note(const std::vector<float>& settings,
                                            float reference_frequency_hz, int midi_note,
                                            bool pitch_track_osc_a, bool pitch_track_osc_b) {
    std::vector<float> retuned = settings;
    if (retuned.size() < constants::NUM_SETTINGS) {
        retuned.resize(constants::NUM_SETTINGS, 0.5f);
    }

    const float safe_reference =
        std::max(1.0f, reference_frequency_hz > 0.0f ? reference_frequency_hz
                                                     : kDefaultReferenceFrequencyHz);
    const float note_hz =
        440.0f * std::pow(2.0f, (static_cast<float>(midi_note) - 69.0f) / 12.0f);
    const float semitone_offset = 12.0f * std::log2(note_hz / safe_reference);
    const float ratio = std::pow(2.0f, semitone_offset / 12.0f);

    if (pitch_track_osc_a) {
        const std::array<int, 2> osc_a_indices = {constants::OSC_A_FREQ_LOW_INDEX,
                                                  constants::OSC_A_FREQ_HIGH_INDEX};
        for (int index : osc_a_indices) {
            const auto current_hz = normalized_frequency_to_hz(retuned[static_cast<size_t>(index)]);
            retuned[static_cast<size_t>(index)] = hz_to_normalized_frequency(current_hz * ratio);
        }
    }

    if (pitch_track_osc_b) {
        const std::array<int, 2> osc_b_indices = {constants::OSC_B_FREQ_LOW_INDEX,
                                                  constants::OSC_B_FREQ_HIGH_INDEX};
        for (int index : osc_b_indices) {
            const auto current_hz = normalized_frequency_to_hz(retuned[static_cast<size_t>(index)]);
            retuned[static_cast<size_t>(index)] = hz_to_normalized_frequency(current_hz * ratio);
        }
    }

    return retuned;
}

std::vector<float> render_note_audio(const std::vector<float>& settings,
                                     float reference_frequency_hz, int midi_note,
                                     double output_sample_rate, bool pitch_track_osc_a,
                                     bool pitch_track_osc_b) {
    const auto duration_seconds = max_envelope_duration_seconds(settings);
    const auto time = make_time_axis(duration_seconds, output_sample_rate);
    auto note_settings = retune_settings_for_note(settings, reference_frequency_hz, midi_note,
                                                  pitch_track_osc_a, pitch_track_osc_b);
    return synth(note_settings, time, static_cast<float>(output_sample_rate));
}

TrainingResult train_synth_with_coarse_options(const std::vector<float>& target_audio,
                                               const CoarseSearchOptions& coarse_options,
                                               int population_size, float initial_sigma,
                                               float time_limit, bool verbose,
                                               TrainingProgressCallback progress_callback,
                                               uint32_t loss_seed) {
    const auto global_start = std::chrono::steady_clock::now();
    const auto time = make_time_axis(constants::NUM_SECONDS, constants::TRAINING_SAMPLE_RATE);
    LossFunction<float> full_loss_fn(target_audio, make_loss_seed(loss_seed));
    const auto summary = summarize_target_audio(target_audio, full_loss_fn.features());
    const auto summary_seed = make_seed_from_summary(summary);
    const int coarse_population =
        population_size > 0 ? population_size : kDefaultCRFMNESPopulationSize;
    const int refinement_population = kRefinementPopulationSize;
    const float effective_sigma = initial_sigma;
    const float coarse_time_budget =
        time_limit > 0.0f ? std::max(0.35f, time_limit * std::clamp(coarse_options.time_fraction, 0.1f, 0.85f))
                          : 1.0f;

    auto coarse_result =
        run_coarse_search(summary_seed, time, full_loss_fn, coarse_population, coarse_options,
                          coarse_time_budget, effective_sigma, progress_callback);
    const float coarse_elapsed = elapsed_seconds_since(global_start);
    report_progress(progress_callback, 0, 0, 0.0f, time_limit, coarse_result.full_loss,
                    TrainingProgress {coarse_result.iterations_completed, coarse_result.full_loss,
                                      effective_sigma, coarse_result.eval_count, coarse_elapsed});

    const float remaining_time =
        time_limit > 0.0f ? std::max(0.1f, time_limit - coarse_elapsed) : time_limit;
    if (time_limit > 0.0f && remaining_time <= 0.15f) {
        TrainingResult result;
        result.best_settings = coarse_result.settings;
        result.best_loss = coarse_result.full_loss;
        result.iterations_completed = 1;
        result.final_eval_count = coarse_result.eval_count;
        result.final_sigma = effective_sigma;
        return result;
    }

    auto synth_fn = [](const std::vector<float>& settings, const std::vector<float>& local_time) {
        return synth(settings, local_time, static_cast<float>(constants::TRAINING_SAMPLE_RATE));
    };

    CRFMNESOptions<float> options;
    options.initial_settings = coarse_result.settings;

    auto nes_progress = [&](const CRFMNESProgress<float>& progress) {
        report_progress(progress_callback, coarse_result.iterations_completed,
                        coarse_result.eval_count, coarse_elapsed, time_limit,
                        std::min(coarse_result.full_loss, progress.best_loss), progress);
    };

    auto result =
        run_crfmnes_optimization<float>(full_loss_fn, time, synth_fn, refinement_population,
                                        effective_sigma, remaining_time, 10000, verbose,
                                        nes_progress, options);

    if (result.best_settings.empty()) {
        result.best_settings = coarse_result.settings;
        result.best_loss = coarse_result.full_loss;
    }

    if (coarse_result.full_loss < result.best_loss) {
        result.best_settings = coarse_result.settings;
        result.best_loss = coarse_result.full_loss;
    }

    result.iterations_completed += coarse_result.iterations_completed;
    result.final_eval_count += coarse_result.eval_count;

    return result;
}

TrainingResult train_synth(const std::vector<float>& target_audio, int population_size,
                           float initial_sigma, float time_limit, bool verbose,
                           TrainingProgressCallback progress_callback, uint32_t loss_seed) {
    return train_synth_with_coarse_options(target_audio, CoarseSearchOptions {}, population_size,
                                           initial_sigma, time_limit, verbose,
                                           std::move(progress_callback), loss_seed);
}

std::string serialize_settings(const std::vector<float>& settings) {
    std::ostringstream stream;
    for (size_t i = 0; i < settings.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << settings[i];
    }
    return stream.str();
}

std::vector<float> deserialize_settings(const std::string& text) {
    auto settings = default_settings();
    std::istringstream stream(text);
    std::string token;
    size_t index = 0;
    while (std::getline(stream, token, ',') && index < settings.size()) {
        if (!token.empty()) {
            settings[index] = std::stof(token);
        }
        ++index;
    }
    return settings;
}

}  // namespace adaptive_echo
