#pragma once

#include <pocketfft_hdronly.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "adaptive_echo/constants.hpp"

namespace adaptive_echo {

enum class RandomWindowType {
    Hann,
    Hamming,
    Blackman,
    Kaiser,
    Tukey,
    Gaussian,
};

enum class RandomCepstralFeatureFamily {
    FullSpectrum,
    LogBands,
};

template <typename T>
struct RandomWindowSpec {
    size_t fft_size = 0;
    size_t offset = 0;
    RandomWindowType window_type = RandomWindowType::Hann;
    T parameter = static_cast<T>(0);
};

template <typename T>
struct RandomCepstralFeatureSpec {
    RandomCepstralFeatureFamily family = RandomCepstralFeatureFamily::FullSpectrum;
    size_t coefficient = 0;
    size_t band_count = 0;
    T weight = static_cast<T>(1);
};

template <typename T>
struct RandomLossWindow {
    RandomWindowSpec<T> window;
    std::vector<RandomCepstralFeatureSpec<T>> features;
};

template <typename T>
struct RandomLossSchedule {
    std::vector<RandomLossWindow<T>> windows;
};

template <typename T>
struct TargetFeaturesFast {
    std::vector<T> target_audio;
    std::vector<size_t> fft_sizes;
    std::vector<size_t> num_freqs;
    std::vector<size_t> num_frames;
    std::vector<std::vector<T>> stfts;
};

namespace detail {

inline constexpr size_t kRandomWindowCount = 128;
inline constexpr size_t kRandomFeaturesPerWindow = 32;
inline constexpr size_t kFullSpectrumCoeffLimit = 64;
inline constexpr size_t kSpectralGroupCount16 = 16;
inline constexpr size_t kSpectralGroupCount4 = 4;
inline constexpr std::array<size_t, 4> kFftCandidates = {256, 512, 1024, 2048};

template <typename T>
inline T clamp_epsilon(T value, T epsilon = static_cast<T>(1e-8)) {
    return std::max(value, epsilon);
}

template <typename T>
inline uint64_t mix_seed(T base_seed, uint64_t counter) {
    uint64_t x = (static_cast<uint64_t>(base_seed) << 32) ^ counter ^ 0x9e3779b97f4a7c15ULL;
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

template <typename T>
inline std::vector<size_t> valid_fft_sizes(size_t signal_length) {
    std::vector<size_t> sizes;
    sizes.reserve(kFftCandidates.size());
    const size_t min_fft = kFftCandidates.front();
    const size_t effective_length = std::max(signal_length, min_fft);
    for (size_t fft_size : kFftCandidates) {
        if (fft_size <= effective_length) {
            sizes.push_back(fft_size);
        }
    }
    if (sizes.empty()) {
        sizes.push_back(min_fft);
    }
    return sizes;
}

template <typename T>
inline std::vector<T> compute_frequency_weights(size_t num_freqs, size_t n_fft, T sample_rate) {
    std::vector<T> weights(num_freqs, static_cast<T>(1));
    const T f_s = sample_rate;
    const T n_f = static_cast<T>(n_fft);

    for (size_t i = 0; i < num_freqs; ++i) {
        const T freq = static_cast<T>(i) * f_s / n_f;
        const T f2 = freq * freq;
        const T f4 = f2 * f2;
        const T c1 = static_cast<T>(12194.217 * 12194.217);
        const T c2 = static_cast<T>(20.601103 * 20.601103);
        const T c3 = static_cast<T>(107.65265 * 107.65265);
        const T c4 = static_cast<T>(737.86223 * 737.86223);
        const T numerator = c1 * f4;
        const T denominator =
            (f2 + c2) * std::sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1) + static_cast<T>(1e-8);
        const T a_weight = numerator / denominator;
        const T mel_slope =
            static_cast<T>(1) / (static_cast<T>(1) + freq / static_cast<T>(700));
        weights[i] = a_weight * mel_slope;
    }

    const T sum = std::accumulate(weights.begin(), weights.end(), static_cast<T>(0));
    if (sum > static_cast<T>(0)) {
        const T scale = static_cast<T>(num_freqs) / sum;
        for (T& weight : weights) {
            weight *= scale;
        }
    }

    return weights;
}

template <typename T>
inline std::vector<size_t> compute_log_band_edges(size_t num_freqs, size_t num_bands) {
    if (num_freqs == 0) {
        return {0};
    }

    num_bands = std::max<size_t>(1, std::min(num_bands, num_freqs));
    std::vector<size_t> edges(num_bands + 1);
    edges.front() = 0;
    edges.back() = num_freqs;
    if (num_bands == 1) {
        return edges;
    }

    const T min_bin = static_cast<T>(1);
    const T max_bin = static_cast<T>(std::max<size_t>(1, num_freqs - 1));
    for (size_t band = 1; band < num_bands; ++band) {
        const T alpha = static_cast<T>(band) / static_cast<T>(num_bands);
        const T log_bin = std::exp(std::log(min_bin) +
                                   alpha * (std::log(max_bin) - std::log(min_bin)));
        edges[band] = static_cast<size_t>(std::round(log_bin));
        edges[band] = std::clamp(edges[band], edges[band - 1] + 1, num_freqs - (num_bands - band));
    }
    return edges;
}

template <typename T>
inline std::vector<T> aggregate_band_weights(const std::vector<T>& freq_weights,
                                             const std::vector<size_t>& band_edges) {
    const size_t num_bands = band_edges.size() > 1 ? band_edges.size() - 1 : 0;
    std::vector<T> band_weights(num_bands, static_cast<T>(0));
    for (size_t band = 0; band < num_bands; ++band) {
        T sum = static_cast<T>(0);
        for (size_t bin = band_edges[band]; bin < band_edges[band + 1]; ++bin) {
            sum += freq_weights[bin];
        }
        band_weights[band] = sum;
    }

    const T total = std::accumulate(band_weights.begin(), band_weights.end(), static_cast<T>(0));
    if (total > static_cast<T>(0) && num_bands > 0) {
        const T scale = static_cast<T>(num_bands) / total;
        for (T& weight : band_weights) {
            weight *= scale;
        }
    }
    return band_weights;
}

template <typename T>
inline std::vector<T> perceptual_cepstral_weights(const std::vector<T>& source_weights,
                                                  size_t num_coeffs) {
    std::vector<T> weights(num_coeffs, static_cast<T>(0));
    if (source_weights.empty() || num_coeffs == 0) {
        return weights;
    }

    const size_t num_bins = source_weights.size();
    const T dct_scale = static_cast<T>(M_PI) / static_cast<T>(num_bins);
    for (size_t coeff = 0; coeff < num_coeffs; ++coeff) {
        T energy = static_cast<T>(0);
        for (size_t bin = 0; bin < num_bins; ++bin) {
            const T angle =
                dct_scale * (static_cast<T>(bin) + static_cast<T>(0.5)) * static_cast<T>(coeff);
            energy += source_weights[bin] * std::abs(std::cos(angle));
        }
        weights[coeff] = energy / static_cast<T>(num_bins);
    }

    const T total = std::accumulate(weights.begin(), weights.end(), static_cast<T>(0));
    if (total > static_cast<T>(0)) {
        const T scale = static_cast<T>(num_coeffs) / total;
        for (T& weight : weights) {
            weight *= scale;
        }
    }
    return weights;
}

template <typename T>
inline T bessel_i0(T x) {
    const T ax = std::abs(x);
    if (ax < static_cast<T>(3.75)) {
        const T y = (x / static_cast<T>(3.75));
        const T y2 = y * y;
        return static_cast<T>(1.0) +
               y2 * (static_cast<T>(3.5156229) +
                     y2 * (static_cast<T>(3.0899424) +
                           y2 * (static_cast<T>(1.2067492) +
                                 y2 * (static_cast<T>(0.2659732) +
                                       y2 * (static_cast<T>(0.0360768) +
                                             y2 * static_cast<T>(0.0045813))))));
    }

    const T y = static_cast<T>(3.75) / ax;
    const T polynomial =
        static_cast<T>(0.39894228) +
        y * (static_cast<T>(0.01328592) +
             y * (static_cast<T>(0.00225319) +
                  y * (static_cast<T>(-0.00157565) +
                       y * (static_cast<T>(0.00916281) +
                            y * (static_cast<T>(-0.02057706) +
                                 y * (static_cast<T>(0.02635537) +
                                      y * (static_cast<T>(-0.01647633) +
                                           y * static_cast<T>(0.00392377))))))));
    return (std::exp(ax) / std::sqrt(ax)) * polynomial;
}

template <typename T>
inline std::vector<T> make_window(const RandomWindowSpec<T>& spec) {
    std::vector<T> window(spec.fft_size, static_cast<T>(1));
    if (spec.fft_size <= 1) {
        return window;
    }

    const T denom = static_cast<T>(spec.fft_size - 1);
    const T two_pi = static_cast<T>(2) * static_cast<T>(M_PI);
    const T mid = denom / static_cast<T>(2);

    switch (spec.window_type) {
        case RandomWindowType::Hann:
            for (size_t i = 0; i < spec.fft_size; ++i) {
                window[i] = static_cast<T>(0.5) -
                            static_cast<T>(0.5) *
                                std::cos(two_pi * static_cast<T>(i) / denom);
            }
            break;
        case RandomWindowType::Hamming:
            for (size_t i = 0; i < spec.fft_size; ++i) {
                window[i] = static_cast<T>(0.54) -
                            static_cast<T>(0.46) *
                                std::cos(two_pi * static_cast<T>(i) / denom);
            }
            break;
        case RandomWindowType::Blackman:
            for (size_t i = 0; i < spec.fft_size; ++i) {
                const T phase = two_pi * static_cast<T>(i) / denom;
                window[i] = static_cast<T>(0.42) - static_cast<T>(0.5) * std::cos(phase) +
                            static_cast<T>(0.08) * std::cos(static_cast<T>(2) * phase);
            }
            break;
        case RandomWindowType::Kaiser: {
            const T beta = spec.parameter;
            const T denom_i0 = std::max(bessel_i0(beta), static_cast<T>(1e-8));
            for (size_t i = 0; i < spec.fft_size; ++i) {
                const T ratio = (static_cast<T>(2) * static_cast<T>(i) / denom) - static_cast<T>(1);
                const T inside = std::max(static_cast<T>(0), static_cast<T>(1) - ratio * ratio);
                window[i] = bessel_i0(beta * std::sqrt(inside)) / denom_i0;
            }
            break;
        }
        case RandomWindowType::Tukey: {
            const T alpha = std::clamp(spec.parameter, static_cast<T>(0.1), static_cast<T>(0.9));
            for (size_t i = 0; i < spec.fft_size; ++i) {
                const T x = static_cast<T>(i) / denom;
                if (x < alpha / static_cast<T>(2)) {
                    window[i] = static_cast<T>(0.5) *
                                (static_cast<T>(1) +
                                 std::cos(static_cast<T>(M_PI) *
                                          (static_cast<T>(2) * x / alpha - static_cast<T>(1))));
                } else if (x <= static_cast<T>(1) - alpha / static_cast<T>(2)) {
                    window[i] = static_cast<T>(1);
                } else {
                    window[i] = static_cast<T>(0.5) *
                                (static_cast<T>(1) +
                                 std::cos(static_cast<T>(M_PI) *
                                          (static_cast<T>(2) * x / alpha -
                                           static_cast<T>(2) / alpha + static_cast<T>(1))));
                }
            }
            break;
        }
        case RandomWindowType::Gaussian: {
            const T sigma = std::clamp(spec.parameter, static_cast<T>(0.2), static_cast<T>(0.6));
            const T scale = std::max(sigma * mid, static_cast<T>(1e-8));
            for (size_t i = 0; i < spec.fft_size; ++i) {
                const T z = (static_cast<T>(i) - mid) / scale;
                window[i] = std::exp(static_cast<T>(-0.5) * z * z);
            }
            break;
        }
    }

    return window;
}

template <typename T>
inline std::vector<T> compute_windowed_magnitude_spectrum(const std::vector<T>& audio,
                                                          const RandomWindowSpec<T>& spec) {
    using complex_t = std::complex<T>;

    const auto window = make_window(spec);
    std::vector<complex_t> frame(spec.fft_size, complex_t(0, 0));
    std::vector<complex_t> fft_output(spec.fft_size, complex_t(0, 0));
    for (size_t i = 0; i < spec.fft_size; ++i) {
        const size_t index = spec.offset + i;
        const T sample = index < audio.size() ? audio[index] : static_cast<T>(0);
        frame[i] = complex_t(sample * window[i], 0);
    }

    pocketfft::shape_t shape{spec.fft_size};
    pocketfft::stride_t stride_in{sizeof(complex_t)};
    pocketfft::stride_t stride_out{sizeof(complex_t)};
    pocketfft::shape_t axes{0};
    pocketfft::c2c<T>(shape, stride_in, stride_out, axes, true, frame.data(), fft_output.data(),
                      T(1));

    const size_t num_freqs = spec.fft_size / 2 + 1;
    std::vector<T> magnitude(num_freqs, static_cast<T>(0));
    const T window_sum =
        std::max(std::accumulate(window.begin(), window.end(), static_cast<T>(0)),
                 static_cast<T>(1e-8));
    const T scale = static_cast<T>(1) / window_sum;
    for (size_t bin = 0; bin < num_freqs; ++bin) {
        magnitude[bin] = std::abs(fft_output[bin]) * scale;
    }
    return magnitude;
}

template <typename T>
inline std::vector<T> compute_summary_stft(const std::vector<T>& audio, size_t fft_size,
                                           size_t hop, size_t* num_frames_out = nullptr) {
    const size_t num_freqs = fft_size / 2 + 1;
    if (audio.empty()) {
        if (num_frames_out != nullptr) {
            *num_frames_out = 0;
        }
        return {};
    }

    std::vector<RandomWindowSpec<T>> frames;
    if (audio.size() <= fft_size) {
        frames.push_back(RandomWindowSpec<T> {fft_size, 0, RandomWindowType::Hann, static_cast<T>(0)});
    } else {
        hop = std::max<size_t>(1, hop);
        for (size_t offset = 0; offset < audio.size(); offset += hop) {
            frames.push_back(
                RandomWindowSpec<T> {fft_size, std::min(offset, audio.size() - 1), RandomWindowType::Hann,
                                     static_cast<T>(0)});
            if (offset + fft_size >= audio.size()) {
                break;
            }
        }
    }

    std::vector<T> stft(frames.size() * num_freqs, static_cast<T>(0));
    for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
        auto spectrum = compute_windowed_magnitude_spectrum(audio, frames[frame_index]);
        auto destination =
            stft.begin() + static_cast<std::ptrdiff_t>(frame_index * num_freqs);
        std::copy(spectrum.begin(), spectrum.end(), destination);
    }

    if (num_frames_out != nullptr) {
        *num_frames_out = frames.size();
    }
    return stft;
}

template <typename T>
inline std::vector<T> compute_band_spectrum(const std::vector<T>& magnitude,
                                            const std::vector<size_t>& band_edges,
                                            const std::vector<T>& freq_weights) {
    const size_t num_bands = band_edges.size() > 1 ? band_edges.size() - 1 : 0;
    std::vector<T> bands(num_bands, static_cast<T>(0));
    for (size_t band = 0; band < num_bands; ++band) {
        const size_t start = band_edges[band];
        const size_t end = band_edges[band + 1];
        T weighted_sum = static_cast<T>(0);
        T weight_sum = static_cast<T>(0);
        for (size_t bin = start; bin < end; ++bin) {
            weighted_sum += magnitude[bin] * freq_weights[bin];
            weight_sum += freq_weights[bin];
        }
        const size_t width = std::max<size_t>(1, end - start);
        bands[band] = weight_sum > static_cast<T>(0) ? weighted_sum / weight_sum
                                                     : weighted_sum / static_cast<T>(width);
    }
    return bands;
}

template <typename T>
inline std::vector<size_t> compute_linear_group_edges(size_t num_freqs, size_t group_count) {
    if (num_freqs == 0) {
        return {0};
    }

    group_count = std::max<size_t>(1, std::min(group_count, num_freqs));
    std::vector<size_t> edges(group_count + 1);
    for (size_t group = 0; group <= group_count; ++group) {
        edges[group] = group * num_freqs / group_count;
    }
    edges.front() = 0;
    edges.back() = num_freqs;
    return edges;
}

template <typename T>
inline std::vector<T> compute_grouped_magnitude_sums(const std::vector<T>& magnitude,
                                                     const std::vector<size_t>& group_edges) {
    const size_t group_count = group_edges.size() > 1 ? group_edges.size() - 1 : 0;
    std::vector<T> grouped_sums(group_count, static_cast<T>(0));
    for (size_t group = 0; group < group_count; ++group) {
        T sum = static_cast<T>(0);
        for (size_t bin = group_edges[group]; bin < group_edges[group + 1]; ++bin) {
            sum += magnitude[bin];
        }
        grouped_sums[group] = sum;
    }
    return grouped_sums;
}

template <typename T>
inline T compute_weighted_cepstral_coefficient(const std::vector<T>& spectrum,
                                               const std::vector<T>& weights,
                                               size_t coefficient) {
    if (spectrum.empty() || spectrum.size() != weights.size()) {
        return static_cast<T>(0);
    }

    const size_t count = spectrum.size();
    const T dct_scale = static_cast<T>(M_PI) / static_cast<T>(count);
    T sum = static_cast<T>(0);
    for (size_t i = 0; i < count; ++i) {
        const T weighted_log = std::log(clamp_epsilon(spectrum[i] * weights[i]));
        const T angle =
            dct_scale * (static_cast<T>(i) + static_cast<T>(0.5)) * static_cast<T>(coefficient);
        sum += weighted_log * std::cos(angle);
    }
    return sum / static_cast<T>(count);
}

template <typename T>
inline T compute_weighted_log_magnitude_loss(const std::vector<T>& generated_spectrum,
                                             const std::vector<T>& target_spectrum,
                                             const std::vector<T>& frequency_weights) {
    if (generated_spectrum.empty() || generated_spectrum.size() != target_spectrum.size() ||
        generated_spectrum.size() != frequency_weights.size()) {
        return static_cast<T>(0);
    }

    constexpr T epsilon = static_cast<T>(1e-8);
    T loss = static_cast<T>(0);
    for (size_t bin = 0; bin < generated_spectrum.size(); ++bin) {
        const T generated_log = std::log(generated_spectrum[bin] + epsilon);
        const T target_log = std::log(target_spectrum[bin] + epsilon);
        loss += frequency_weights[bin] * std::abs(generated_log - target_log);
    }

    return loss / static_cast<T>(generated_spectrum.size());
}

template <typename T>
inline T compute_centered_log_structure_scale(const std::vector<T>& spectrum,
                                              const std::vector<T>& frequency_weights) {
    if (spectrum.empty() || spectrum.size() != frequency_weights.size()) {
        return static_cast<T>(0);
    }

    constexpr T epsilon = static_cast<T>(1e-8);
    std::vector<T> log_spectrum(spectrum.size(), static_cast<T>(0));
    T mean_log = static_cast<T>(0);
    for (size_t bin = 0; bin < spectrum.size(); ++bin) {
        log_spectrum[bin] = std::log(spectrum[bin] + epsilon);
        mean_log += log_spectrum[bin];
    }
    mean_log /= static_cast<T>(spectrum.size());

    T scale = static_cast<T>(0);
    for (size_t bin = 0; bin < spectrum.size(); ++bin) {
        scale += frequency_weights[bin] * std::abs(log_spectrum[bin] - mean_log);
    }
    return scale / static_cast<T>(spectrum.size());
}

template <typename T>
inline RandomWindowType sample_window_type(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> distribution(0, 5);
    return static_cast<RandomWindowType>(distribution(rng));
}

template <typename T>
inline RandomWindowSpec<T> sample_window_spec(size_t target_length, std::mt19937_64& rng) {
    const auto sizes = valid_fft_sizes<T>(target_length);
    std::uniform_int_distribution<size_t> fft_distribution(0, sizes.size() - 1);
    RandomWindowSpec<T> spec;
    spec.fft_size = sizes[fft_distribution(rng)];
    spec.window_type = sample_window_type<T>(rng);

    const size_t max_offset = target_length > spec.fft_size ? target_length - spec.fft_size : 0;
    std::uniform_int_distribution<size_t> offset_distribution(0, max_offset);
    spec.offset = max_offset > 0 ? offset_distribution(rng) : 0;

    std::uniform_real_distribution<T> unit_distribution(static_cast<T>(0), static_cast<T>(1));
    switch (spec.window_type) {
        case RandomWindowType::Hann:
        case RandomWindowType::Hamming:
        case RandomWindowType::Blackman:
            spec.parameter = static_cast<T>(0);
            break;
        case RandomWindowType::Kaiser:
            spec.parameter = static_cast<T>(4) + unit_distribution(rng) * static_cast<T>(10);
            break;
        case RandomWindowType::Tukey:
            spec.parameter = static_cast<T>(0.1) + unit_distribution(rng) * static_cast<T>(0.8);
            break;
        case RandomWindowType::Gaussian:
            spec.parameter = static_cast<T>(0.2) + unit_distribution(rng) * static_cast<T>(0.4);
            break;
        default:
            spec.parameter = static_cast<T>(0);
            break;
    }

    return spec;
}

template <typename T>
inline RandomLossSchedule<T> make_random_loss_schedule(const std::vector<T>& target_audio,
                                                       uint64_t schedule_seed) {
    std::mt19937_64 rng(schedule_seed);
    RandomLossSchedule<T> schedule;
    schedule.windows.reserve(kRandomWindowCount);

    for (size_t window_index = 0; window_index < kRandomWindowCount; ++window_index) {
        RandomLossWindow<T> loss_window;
        loss_window.window = sample_window_spec<T>(target_audio.size(), rng);
        loss_window.features.reserve(kRandomFeaturesPerWindow);

        const size_t num_freqs = loss_window.window.fft_size / 2 + 1;
        const size_t full_coeffs = std::max<size_t>(2, std::min(kFullSpectrumCoeffLimit, num_freqs));
        std::bernoulli_distribution family_distribution(0.5);
        std::uniform_int_distribution<size_t> full_coeff_distribution(1, full_coeffs - 1);
        std::uniform_int_distribution<size_t> band_count_distribution(
            8, std::min<size_t>(32, std::max<size_t>(8, num_freqs)));

        for (size_t feature_index = 0; feature_index < kRandomFeaturesPerWindow; ++feature_index) {
            RandomCepstralFeatureSpec<T> feature;
            const bool use_bands = family_distribution(rng);
            if (!use_bands) {
                feature.family = RandomCepstralFeatureFamily::FullSpectrum;
                feature.coefficient = full_coeff_distribution(rng);
                feature.band_count = 0;
            } else {
                feature.family = RandomCepstralFeatureFamily::LogBands;
                feature.band_count = band_count_distribution(rng);
                const size_t clamped_band_count = std::min(feature.band_count, num_freqs);
                std::uniform_int_distribution<size_t> band_coeff_distribution(
                    1, std::max<size_t>(1, clamped_band_count - 1));
                feature.coefficient = band_coeff_distribution(rng);
                feature.band_count = clamped_band_count;
            }
            loss_window.features.push_back(feature);
        }

        schedule.windows.push_back(std::move(loss_window));
    }

    return schedule;
}

template <typename T>
struct PreparedBandContext {
    size_t band_count = 0;
    std::vector<size_t> band_edges;
    std::vector<T> band_weights;
    std::vector<T> cepstral_weights;
    std::vector<T> target_band_spectrum;
};

template <typename T>
struct PreparedFeature {
    RandomCepstralFeatureFamily family = RandomCepstralFeatureFamily::FullSpectrum;
    size_t coefficient = 0;
    size_t band_count = 0;
    T weight = static_cast<T>(1);
    T target_value = static_cast<T>(0);
};

template <typename T>
struct PreparedWindowContext {
    RandomWindowSpec<T> window;
    std::vector<T> frequency_weights;
    std::vector<T> target_spectrum;
    std::vector<size_t> spectral_group_edges_16;
    std::vector<T> spectral_group_weights_16;
    std::vector<T> target_grouped_spectrum_16;
    std::vector<size_t> spectral_group_edges_4;
    std::vector<T> spectral_group_weights_4;
    std::vector<T> target_grouped_spectrum_4;
    std::vector<T> full_cepstral_weights;
    T spectral_normalization = static_cast<T>(1);
    std::vector<PreparedBandContext<T>> band_contexts;
    std::vector<PreparedFeature<T>> features;
};

template <typename T>
inline PreparedBandContext<T>* find_band_context(std::vector<PreparedBandContext<T>>& band_contexts,
                                                 size_t band_count) {
    for (auto& context : band_contexts) {
        if (context.band_count == band_count) {
            return &context;
        }
    }
    return nullptr;
}

template <typename T>
inline const PreparedBandContext<T>* find_band_context(
    const std::vector<PreparedBandContext<T>>& band_contexts, size_t band_count) {
    for (const auto& context : band_contexts) {
        if (context.band_count == band_count) {
            return &context;
        }
    }
    return nullptr;
}

template <typename T>
inline PreparedBandContext<T> make_band_context(const std::vector<T>& target_spectrum,
                                                const std::vector<T>& frequency_weights,
                                                size_t band_count) {
    PreparedBandContext<T> context;
    context.band_count = band_count;
    context.band_edges = compute_log_band_edges<T>(target_spectrum.size(), band_count);
    context.band_weights = aggregate_band_weights(frequency_weights, context.band_edges);
    context.cepstral_weights = perceptual_cepstral_weights(context.band_weights, band_count);
    context.target_band_spectrum =
        compute_band_spectrum(target_spectrum, context.band_edges, frequency_weights);
    return context;
}

template <typename T>
inline PreparedWindowContext<T> prepare_window_context(const std::vector<T>& target_audio,
                                                       const RandomLossWindow<T>& loss_window) {
    PreparedWindowContext<T> context;
    context.window = loss_window.window;
    context.target_spectrum = compute_windowed_magnitude_spectrum(target_audio, loss_window.window);
    context.frequency_weights = compute_frequency_weights<T>(
        context.target_spectrum.size(), loss_window.window.fft_size,
        static_cast<T>(constants::TRAINING_SAMPLE_RATE));
    context.spectral_group_edges_16 =
        compute_linear_group_edges<T>(context.target_spectrum.size(), kSpectralGroupCount16);
    context.spectral_group_weights_16 =
        aggregate_band_weights(context.frequency_weights, context.spectral_group_edges_16);
    context.target_grouped_spectrum_16 =
        compute_grouped_magnitude_sums(context.target_spectrum, context.spectral_group_edges_16);
    context.spectral_group_edges_4 =
        compute_linear_group_edges<T>(context.target_spectrum.size(), kSpectralGroupCount4);
    context.spectral_group_weights_4 =
        aggregate_band_weights(context.frequency_weights, context.spectral_group_edges_4);
    context.target_grouped_spectrum_4 =
        compute_grouped_magnitude_sums(context.target_spectrum, context.spectral_group_edges_4);

    const size_t full_coeff_count =
        std::max<size_t>(2, std::min(kFullSpectrumCoeffLimit, context.target_spectrum.size()));
    context.full_cepstral_weights =
        perceptual_cepstral_weights(context.frequency_weights, full_coeff_count);
    context.features.reserve(loss_window.features.size());
    T cepstral_reference = static_cast<T>(0);

    for (const auto& feature_spec : loss_window.features) {
        PreparedFeature<T> feature;
        feature.family = feature_spec.family;
        feature.coefficient = feature_spec.coefficient;
        feature.band_count = feature_spec.band_count;

        if (feature.family == RandomCepstralFeatureFamily::FullSpectrum) {
            const size_t coeff_index =
                std::min(feature.coefficient, context.full_cepstral_weights.size() - 1);
            feature.coefficient = coeff_index;
            feature.weight = context.full_cepstral_weights[coeff_index];
            feature.target_value = compute_weighted_cepstral_coefficient(
                context.target_spectrum, context.frequency_weights, coeff_index);
        } else {
            PreparedBandContext<T>* band_context =
                find_band_context(context.band_contexts, feature.band_count);
            if (band_context == nullptr) {
                context.band_contexts.push_back(make_band_context(
                    context.target_spectrum, context.frequency_weights, feature.band_count));
                band_context = &context.band_contexts.back();
            }
            const size_t coeff_index =
                std::min(feature.coefficient, band_context->cepstral_weights.size() - 1);
            feature.coefficient = coeff_index;
            feature.weight = band_context->cepstral_weights[coeff_index];
            feature.target_value = compute_weighted_cepstral_coefficient(
                band_context->target_band_spectrum, band_context->band_weights, coeff_index);
        }

        cepstral_reference += feature.weight * std::abs(feature.target_value);
        context.features.push_back(feature);
    }

    const T mean_cepstral_reference =
        context.features.empty()
            ? static_cast<T>(0)
            : cepstral_reference / static_cast<T>(context.features.size());
    const T spectral_reference =
        compute_centered_log_structure_scale(context.target_spectrum, context.frequency_weights);
    constexpr T epsilon = static_cast<T>(1e-8);
    if (spectral_reference > epsilon && mean_cepstral_reference > epsilon) {
        context.spectral_normalization = spectral_reference / mean_cepstral_reference;
    }

    return context;
}

template <typename T>
inline std::vector<PreparedWindowContext<T>> prepare_schedule(
    const TargetFeaturesFast<T>& target_features, const RandomLossSchedule<T>& schedule) {
    std::vector<PreparedWindowContext<T>> contexts;
    contexts.reserve(schedule.windows.size());
    for (const auto& window : schedule.windows) {
        contexts.push_back(prepare_window_context(target_features.target_audio, window));
    }
    return contexts;
}

template <typename T>
inline T evaluate_loss_for_signal(const std::vector<T>& generated,
                                  const std::vector<PreparedWindowContext<T>>& contexts) {
    if (contexts.empty()) {
        return static_cast<T>(0);
    }

    T cepstral_loss = static_cast<T>(0);
    T spectral_loss = static_cast<T>(0);
    size_t total_cepstral_terms = 0;
    size_t total_spectral_terms = 0;
    for (const auto& context : contexts) {
        const auto spectrum = compute_windowed_magnitude_spectrum(generated, context.window);
        const auto grouped_spectrum_16 =
            compute_grouped_magnitude_sums(spectrum, context.spectral_group_edges_16);
        const auto grouped_spectrum_4 =
            compute_grouped_magnitude_sums(spectrum, context.spectral_group_edges_4);
        const T full_spectral_term = compute_weighted_log_magnitude_loss(
            spectrum, context.target_spectrum, context.frequency_weights);
        const T grouped_spectral_term_16 = compute_weighted_log_magnitude_loss(
            grouped_spectrum_16, context.target_grouped_spectrum_16, context.spectral_group_weights_16);
        const T grouped_spectral_term_4 = compute_weighted_log_magnitude_loss(
            grouped_spectrum_4, context.target_grouped_spectrum_4, context.spectral_group_weights_4);
        spectral_loss += (full_spectral_term + grouped_spectral_term_16 + grouped_spectral_term_4) /
                         (static_cast<T>(3) * context.spectral_normalization);
        ++total_spectral_terms;
        std::vector<std::pair<size_t, std::vector<T>>> generated_band_cache;
        generated_band_cache.reserve(context.band_contexts.size());

        for (const auto& feature : context.features) {
            T generated_value = static_cast<T>(0);
            if (feature.family == RandomCepstralFeatureFamily::FullSpectrum) {
                generated_value = compute_weighted_cepstral_coefficient(
                    spectrum, context.frequency_weights, feature.coefficient);
            } else {
                auto cache_it = std::find_if(
                    generated_band_cache.begin(), generated_band_cache.end(),
                    [&](const auto& entry) { return entry.first == feature.band_count; });
                if (cache_it == generated_band_cache.end()) {
                    const auto* band_context =
                        find_band_context(context.band_contexts, feature.band_count);
                    if (band_context == nullptr) {
                        continue;
                    }
                    generated_band_cache.emplace_back(
                        feature.band_count,
                        compute_band_spectrum(spectrum, band_context->band_edges,
                                              context.frequency_weights));
                    cache_it = std::prev(generated_band_cache.end());
                }

                const auto* band_context =
                    find_band_context(context.band_contexts, feature.band_count);
                if (band_context == nullptr) {
                    continue;
                }
                generated_value = compute_weighted_cepstral_coefficient(
                    cache_it->second, band_context->band_weights, feature.coefficient);
            }

            cepstral_loss += feature.weight * std::abs(generated_value - feature.target_value);
            ++total_cepstral_terms;
        }
    }

    const T mean_cepstral_loss =
        total_cepstral_terms > 0 ? cepstral_loss / static_cast<T>(total_cepstral_terms)
                                 : static_cast<T>(0);
    const T mean_spectral_loss =
        total_spectral_terms > 0 ? spectral_loss / static_cast<T>(total_spectral_terms)
                                 : static_cast<T>(0);
    return static_cast<T>(0.5) * mean_cepstral_loss +
           static_cast<T>(0.5) * mean_spectral_loss;
}

template <typename T>
inline TargetFeaturesFast<T> precompute_target_features_fast(
    const std::vector<T>& target, const std::vector<size_t>& fft_sizes = {2048}) {
    TargetFeaturesFast<T> features;
    features.target_audio = target;
    features.fft_sizes = fft_sizes;
    features.num_freqs.resize(fft_sizes.size());
    features.num_frames.resize(fft_sizes.size());
    features.stfts.resize(fft_sizes.size());

    for (size_t i = 0; i < fft_sizes.size(); ++i) {
        const size_t fft_size = fft_sizes[i];
        const size_t hop = std::max<size_t>(1, fft_size / 4);
        features.num_freqs[i] = fft_size / 2 + 1;
        features.stfts[i] = compute_summary_stft(target, fft_size, hop, &features.num_frames[i]);
    }

    return features;
}

}  // namespace detail

template <typename T>
inline TargetFeaturesFast<T> precompute_target_features_fast(
    const std::vector<T>& target, const std::vector<size_t>& fft_sizes = {2048}) {
    return detail::precompute_target_features_fast(target, fft_sizes);
}

template <typename T>
inline std::vector<T> compute_audio_loss_batch(const std::vector<std::vector<T>>& generated_batch,
                                               const TargetFeaturesFast<T>& target_features,
                                               uint64_t schedule_seed) {
    const auto schedule = detail::make_random_loss_schedule(target_features.target_audio, schedule_seed);
    const auto prepared = detail::prepare_schedule(target_features, schedule);

    std::vector<T> losses(generated_batch.size(), static_cast<T>(0));
    for (size_t i = 0; i < generated_batch.size(); ++i) {
        losses[i] = detail::evaluate_loss_for_signal(generated_batch[i], prepared);
    }
    return losses;
}

template <typename T>
inline std::vector<T> compute_audio_loss_batch(const std::vector<std::vector<T>>& generated_batch,
                                               const TargetFeaturesFast<T>& target_features) {
    return compute_audio_loss_batch(generated_batch, target_features, 0);
}

template <typename T>
class LossFunction {
   public:
    LossFunction(const std::vector<T>& target, uint32_t base_seed = 0,
                 const std::vector<size_t>& fft_sizes = {2048})
        : base_seed_(base_seed), features_(precompute_target_features_fast(target, fft_sizes)) {}

    T operator()(const std::vector<T>& generated) const {
        const uint64_t schedule_seed =
            detail::mix_seed(base_seed_, schedule_counter_.fetch_add(1, std::memory_order_relaxed));
        return compute_audio_loss_batch<T>({generated}, features_, schedule_seed).front();
    }

    std::vector<T> compute_batch(const std::vector<std::vector<T>>& generated_batch) const {
        const uint64_t schedule_seed =
            detail::mix_seed(base_seed_, schedule_counter_.fetch_add(1, std::memory_order_relaxed));
        return compute_audio_loss_batch(generated_batch, features_, schedule_seed);
    }

    const TargetFeaturesFast<T>& features() const { return features_; }

   private:
    uint32_t base_seed_ = 0;
    mutable std::atomic<uint64_t> schedule_counter_ {0};
    TargetFeaturesFast<T> features_;
};

}  // namespace adaptive_echo
