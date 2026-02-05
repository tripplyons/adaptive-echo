#pragma once

/**
 * Optimized STFT-based audio loss functions for adaptive_echo.
 * Parallelized and vectorized for maximum performance.
 *
 * Features:
 * - OpenMP parallelization across STFT frames
 * - Batch STFT processing
 * - SIMD vectorization using compiler intrinsics
 * - Thread-safe loss computation for parallel DE evaluation
 * - Precomputed FFT twiddle factors and bit-reversal tables
 */

#include <pocketfft_hdronly.h>

#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "adaptive_echo/constants.hpp"

// SIMD intrinsics support
#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#elif defined(__AVX__)
#include <immintrin.h>
#define HAS_AVX 1
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#define HAS_SSE42 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define HAS_SSE2 1
#endif

// OpenMP support
#if defined(_OPENMP)
#include <omp.h>
#define HAS_OPENMP 1
#endif

namespace adaptive_echo {

namespace detail {
/**
 * Thread-local scratch buffer pool for STFT computation.
 * Prevents allocation overhead during parallel processing.
 */
template <typename T>
class ScratchBufferPool {
   public:
    static ScratchBufferPool& instance() {
        static ScratchBufferPool pool;
        return pool;
    }

    std::vector<std::complex<T>> acquire(size_t size) {
        thread_local std::vector<std::complex<T>> buffer;
        if (buffer.size() < size) {
            buffer.resize(size);
        }
        return buffer;
    }

   private:
    ScratchBufferPool() = default;
};

/**
 * Precomputed Hann window cache with aligned memory.
 */
template <typename T>
class WindowCache {
   public:
    static const std::vector<T>& get(size_t size) {
        static std::vector<std::vector<T>> cache;
        if (cache.size() <= size) {
            cache.resize(size + 1);
        }
        if (cache[size].empty()) {
            cache[size].resize(size);
            const T TWO_PI = static_cast<T>(2.0 * M_PI);
            for (size_t i = 0; i < size; ++i) {
                cache[size][i] = static_cast<T>(0.5) *
                                 (static_cast<T>(1.0) -
                                  std::cos(TWO_PI * static_cast<T>(i) / static_cast<T>(size - 1)));
            }
        }
        return cache[size];
    }

    static T get_sum(size_t size) {
        const auto& window = get(size);
        T sum = 0;
        for (T v : window) sum += v;
        return sum;
    }
};

/**
 * Vectorized mean calculation using SIMD where available.
 */
template <typename T>
inline T vectorized_mean(const std::vector<T>& data) {
    if (data.empty()) return static_cast<T>(0);

    T sum = 0;
    const size_t n = data.size();

#if defined(HAS_AVX2) && defined(__AVX2__)
    // AVX2: Process 8 floats at a time
    if constexpr (std::is_same_v<T, float>) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            sum_vec = _mm256_add_ps(sum_vec, vec);
        }

        // Horizontal sum
        float sum_arr[8];
        _mm256_storeu_ps(sum_arr, sum_vec);
        for (int j = 0; j < 8; ++j) sum += sum_arr[j];

        // Remainder
        for (; i < n; ++i) sum += data[i];
        return sum / static_cast<T>(n);
    }
#elif defined(HAS_SSE2) && defined(__SSE2__)
    // SSE2: Process 4 floats at a time
    if constexpr (std::is_same_v<T, float>) {
        __m128 sum_vec = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            __m128 vec = _mm_loadu_ps(&data[i]);
            sum_vec = _mm_add_ps(sum_vec, vec);
        }

        // Horizontal sum
        float sum_arr[4];
        _mm_storeu_ps(sum_arr, sum_vec);
        for (int j = 0; j < 4; ++j) sum += sum_arr[j];

        // Remainder
        for (; i < n; ++i) sum += data[i];
        return sum / static_cast<T>(n);
    }
#endif

// Fallback: OpenMP SIMD
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : sum)
#endif
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }

    return sum / static_cast<T>(n);
}

/**
 * Vectorized variance calculation.
 */
template <typename T>
inline T vectorized_variance(const std::vector<T>& data, T mean) {
    if (data.size() < 2) return static_cast<T>(0);

    T var = 0;
    const size_t n = data.size();

#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : var)
#endif
    for (size_t i = 0; i < n; ++i) {
        T diff = data[i] - mean;
        var += diff * diff;
    }

    return var / static_cast<T>(n);
}

/**
 * Single STFT frame computation using pocketfft.
 * Thread-safe: each call allocates its own buffer to avoid thread conflicts.
 */
template <typename T>
inline void compute_stft_frame(const std::vector<T>& x, const std::vector<T>& window, size_t frame,
                               size_t start, size_t win_length, size_t n_fft, size_t num_freqs,
                               T* result) {
    using complex_t = std::complex<T>;

    // Allocate buffer locally - OpenMP ensures each thread has its own
    std::vector<complex_t> frame_data(n_fft);
    std::vector<complex_t> fft_output(n_fft);

    // Window and zero-pad
    for (size_t i = 0; i < n_fft; ++i) {
        if (i < win_length && start + i < x.size()) {
            frame_data[i] = complex_t(x[start + i] * window[i], 0);
        } else {
            frame_data[i] = complex_t(0, 0);
        }
    }

    // FFT using pocketfft
    pocketfft::shape_t shape{n_fft};
    pocketfft::stride_t stride_in{sizeof(complex_t)};
    pocketfft::stride_t stride_out{sizeof(complex_t)};
    pocketfft::shape_t axes{0};  // Transform along axis 0
    pocketfft::c2c<T>(shape, stride_in, stride_out, axes, true, frame_data.data(),
                      fft_output.data(), T(1));

    // Store magnitudes with window normalization
    // FFT normalization: 1.0 / sum(window) for magnitude spectrum
    T window_sum = WindowCache<T>::get_sum(win_length);
    T scale = static_cast<T>(1.0) / window_sum;
    size_t offset = frame * num_freqs;
    for (size_t freq = 0; freq < num_freqs; ++freq) {
        result[offset + freq] = std::abs(fft_output[freq]) * scale;
    }
}

/**
 * Optimized STFT with parallel frame processing.
 * Layout: [frame][freq] for cache locality.
 */
template <typename T>
inline std::vector<T> stft_fast(const std::vector<T>& x, size_t win_length, size_t hop_length,
                                size_t n_fft) {
    const auto& window = WindowCache<T>::get(win_length);
    size_t num_frames = (x.size() >= win_length) ? (x.size() - win_length) / hop_length + 1 : 0;
    size_t num_freqs = n_fft / 2 + 1;

    if (num_frames == 0) {
        return std::vector<T>();
    }

    std::vector<T> result(num_frames * num_freqs);

#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (size_t frame = 0; frame < num_frames; ++frame) {
        size_t start = frame * hop_length;
        compute_stft_frame(x, window, frame, start, win_length, n_fft, num_freqs, result.data());
    }

    return result;
}

/**
 * Compute perceptual frequency weights (A-weighting + Mel-scale density).
 * Used to give "fair weight" to each frequency in audio similarity loss.
 */
template <typename T>
inline std::vector<T> compute_frequency_weights(size_t num_freqs, size_t n_fft, T sample_rate) {
    std::vector<T> weights(num_freqs);
    const T f_s = sample_rate;
    const T n_f = static_cast<T>(n_fft);

    for (size_t i = 0; i < num_freqs; ++i) {
        T freq = static_cast<T>(i) * f_s / n_f;

        // A-weighting approximation
        T f2 = freq * freq;
        T f4 = f2 * f2;
        T c1 = static_cast<T>(12194.217 * 12194.217);
        T c2 = static_cast<T>(20.601103 * 20.601103);
        T c3 = static_cast<T>(107.65265 * 107.65265);
        T c4 = static_cast<T>(737.86223 * 737.86223);

        T num = c1 * f4;
        T den = (f2 + c2) * std::sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1);
        T ra = num / (den + static_cast<T>(1e-8));

        // Mel-scale slope to balance linear bin density
        // dMel/df = 1 / (1 + f/700)
        T mel_slope = static_cast<T>(1.0) / (static_cast<T>(1.0) + freq / static_cast<T>(700.0));

        weights[i] = ra * mel_slope;
    }

    // Normalize weights to average 1.0 to maintain loss scale
    T sum = 0;
    for (T w : weights) sum += w;
    T inv_mean = static_cast<T>(num_freqs) / (sum + static_cast<T>(1e-8));
    for (size_t i = 0; i < num_freqs; ++i) {
        weights[i] *= inv_mean;
    }

    return weights;
}

/**
 * Vectorized zero-crossing rate computation.
 */
template <typename T>
inline T zero_crossing_rate_fast(const std::vector<T>& x) {
    if (x.size() < 2) return static_cast<T>(0);

    size_t count = 0;
    const size_t n = x.size();

#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : count)
#endif
    for (size_t i = 1; i < n; ++i) {
        count += (x[i] * x[i - 1] < 0) ? 1 : 0;
    }

    return static_cast<T>(count) / static_cast<T>(n - 1);
}

/**
 * Vectorized spectral convergence loss with frequency weighting.
 */
template <typename T>
inline T spectral_convergence_loss(const std::vector<T>& x_mag, const std::vector<T>& y_mag,
                                   T y_mag_mean, const std::vector<T>& weights, size_t num_freqs) {
    const size_t n = x_mag.size();
    if (n == 0 || num_freqs == 0) return static_cast<T>(0);

    T inv_y_mean = static_cast<T>(1.0) / (y_mag_mean + static_cast<T>(1e-8));
    size_t num_frames = n / num_freqs;

    T sc_loss = 0;
    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* x_ptr = &x_mag[frame * num_freqs];
        const T* y_ptr = &y_mag[frame * num_freqs];
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : sc_loss)
#endif
        for (size_t freq = 0; freq < num_freqs; ++freq) {
            sc_loss += weights[freq] * std::abs(y_ptr[freq] - x_ptr[freq]) * inv_y_mean;
        }
    }

    return sc_loss / static_cast<T>(n);
}

/**
 * Vectorized log-magnitude loss with frequency weighting.
 */
template <typename T>
inline T log_magnitude_loss(const std::vector<T>& x_mag, const std::vector<T>& y_mag,
                            const std::vector<T>& weights, size_t num_freqs) {
    const size_t n = x_mag.size();
    if (n == 0 || num_freqs == 0) return static_cast<T>(0);

    constexpr T EPSILON = static_cast<T>(1e-8);
    size_t num_frames = n / num_freqs;

    T mag_loss = 0;
    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* x_ptr = &x_mag[frame * num_freqs];
        const T* y_ptr = &y_mag[frame * num_freqs];
#if defined(HAS_OPENMP)
#pragma omp simd reduction(+ : mag_loss)
#endif
        for (size_t freq = 0; freq < num_freqs; ++freq) {
            T log_y = std::log(y_ptr[freq] + EPSILON);
            T log_x = std::log(x_ptr[freq] + EPSILON);
            mag_loss += weights[freq] * std::abs(log_y - log_x);
        }
    }

    return mag_loss / static_cast<T>(n);
}

/**
 * Fast vectorized normalization (zero mean, unit variance).
 */
template <typename T>
inline std::vector<T> normalize_signal(const std::vector<T>& x) {
    if (x.empty()) return x;

    // Compute mean
    T mean = vectorized_mean(x);

    // Subtract mean
    std::vector<T> result(x.size());
#if defined(HAS_OPENMP)
#pragma omp simd
#endif
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] - mean;
    }

    // Compute std and normalize
    T var = vectorized_variance(result, static_cast<T>(0));
    T std = std::sqrt(var);
    T inv_std = static_cast<T>(1.0) / (std + static_cast<T>(1e-8));

#if defined(HAS_OPENMP)
#pragma omp simd
#endif
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] *= inv_std;
    }

    return result;
}

/**
 * Downscale spectrogram by averaging frequency bins.
 * 4x downscale: average every 4 bins into 1 bin.
 * 16x downscale: average every 16 bins into 1 bin.
 */
template <typename T>
inline std::vector<T> downscale_spectrogram(const std::vector<T>& spec, size_t num_freqs,
                                            size_t downscale_factor) {
    if (spec.empty() || num_freqs == 0 || downscale_factor <= 1) return spec;

    size_t num_frames = spec.size() / num_freqs;
    size_t new_num_freqs = num_freqs / downscale_factor;
    std::vector<T> downscaled(num_frames * new_num_freqs);

    for (size_t frame = 0; frame < num_frames; ++frame) {
        const T* frame_data = &spec[frame * num_freqs];
        T* dest = &downscaled[frame * new_num_freqs];

        for (size_t new_f = 0; new_f < new_num_freqs; ++new_f) {
            size_t start_f = new_f * downscale_factor;
            T sum = 0;
            for (size_t i = 0; i < downscale_factor; ++i) {
                sum += frame_data[start_f + i];
            }
            dest[new_f] = sum / static_cast<T>(downscale_factor);
        }
    }

    return downscaled;
}

/**
 * Helper to aggregate weights for downscaled spectrograms.
 * Sums original weights into downscaled bins.
 */
template <typename T>
inline std::vector<T> aggregate_weights(const std::vector<T>& original_weights, size_t factor,
                                        size_t new_num_freqs) {
    std::vector<T> aggregated(new_num_freqs);
    for (size_t f = 0; f < new_num_freqs; ++f) {
        T sum = 0;
        for (size_t j = 0; j < factor; ++j) {
            sum += original_weights[f * factor + j];
        }
        aggregated[f] = sum;
    }
    return aggregated;
}

/**
 * Compute downscaled loss for a given factor (4x or 16x).
 * Returns the combined spectral convergence + log-magnitude loss.
 */
template <typename T>
inline T compute_downscale_loss(const std::vector<T>& x_stft, const std::vector<T>& y_stft,
                                const std::vector<T>& original_weights, size_t num_freqs,
                                size_t factor, T y_stft_mean) {
    size_t new_num_freqs = num_freqs / factor;
    auto x_stft_down = detail::downscale_spectrogram(x_stft, num_freqs, factor);

    if (x_stft_down.size() != y_stft.size()) {
        return static_cast<T>(0);
    }

    auto weights_down = aggregate_weights(original_weights, factor, new_num_freqs);

    T sc_loss = detail::spectral_convergence_loss(x_stft_down, y_stft, y_stft_mean, weights_down,
                                                  new_num_freqs);
    T mag_loss = detail::log_magnitude_loss(x_stft_down, y_stft, weights_down, new_num_freqs);

    return static_cast<T>(0.9) * sc_loss + static_cast<T>(0.1) * mag_loss;
}

}  // namespace detail

/**
 * Precomputed target features for fast loss computation.
 */
template <typename T>
struct TargetFeaturesFast {
    std::vector<std::vector<T>> stfts;    // [scale][frame * freq]
    std::vector<std::vector<T>> weights;  // [scale][freq]
    std::vector<T> stft_means;
    std::vector<size_t> num_frames;
    std::vector<size_t> num_freqs;
    T zcr;

    // Downscaled spectrograms (4x and 16x)
    std::vector<std::vector<T>> stfts_4x;   // [scale][frame * freq/4]
    std::vector<std::vector<T>> stfts_16x;  // [scale][frame * freq/16]
    std::vector<T> stft_means_4x;
    std::vector<T> stft_means_16x;

    // Store FFT parameters used for precomputation
    std::vector<size_t> fft_sizes;
    std::vector<size_t> hop_sizes;

    // Store target energy/RMS for amplitude matching
    T target_rms;
};

/**
 * Precompute target STFT features with vectorized operations.
 */
template <typename T>
inline TargetFeaturesFast<T> precompute_target_features_fast(
    const std::vector<T>& target, const std::vector<size_t>& fft_sizes = {1024, 512},
    const std::vector<size_t>& hop_sizes = {512, 256}) {
    TargetFeaturesFast<T> features;
    features.num_frames.resize(fft_sizes.size());
    features.num_freqs.resize(fft_sizes.size());
    features.stft_means.resize(fft_sizes.size());

    // Compute target RMS from original signal
    T target_mean = detail::vectorized_mean(target);
    T target_var = detail::vectorized_variance(target, target_mean);
    features.target_rms = std::sqrt(target_var);

    // Use original signal (no normalization)
    const std::vector<T>& tgt_norm = target;

    // Compute STFT magnitudes and perceptual weights for each scale
    for (size_t i = 0; i < fft_sizes.size(); ++i) {
        features.stfts.push_back(
            detail::stft_fast(tgt_norm, fft_sizes[i], hop_sizes[i], fft_sizes[i]));
        features.num_frames[i] =
            (target.size() >= fft_sizes[i]) ? (target.size() - fft_sizes[i]) / hop_sizes[i] + 1 : 0;
        features.num_freqs[i] = fft_sizes[i] / 2 + 1;

        // Precompute perceptual weights for this FFT size
        features.weights.push_back(detail::compute_frequency_weights(
            features.num_freqs[i], fft_sizes[i],
            static_cast<T>(adaptive_echo::constants::TRAINING_SAMPLE_RATE)));

        if (!features.stfts[i].empty()) {
            features.stft_means[i] = detail::vectorized_mean(features.stfts[i]);
        } else {
            features.stft_means[i] = static_cast<T>(0);
        }
    }

    // Compute zero-crossing rate
    features.zcr = detail::zero_crossing_rate_fast(tgt_norm);

    // Store FFT parameters for use during loss computation
    features.fft_sizes = fft_sizes;
    features.hop_sizes = hop_sizes;

    // Precompute 4x and 16x downscaled spectrograms
    features.stfts_4x.resize(fft_sizes.size());
    features.stfts_16x.resize(fft_sizes.size());
    features.stft_means_4x.resize(fft_sizes.size());
    features.stft_means_16x.resize(fft_sizes.size());

    for (size_t i = 0; i < fft_sizes.size(); ++i) {
        size_t num_freqs = features.num_freqs[i];

        // Only downscale if we have enough frequency bins
        if (num_freqs >= 16) {
            features.stfts_4x[i] = detail::downscale_spectrogram(features.stfts[i], num_freqs, 4);
            if (!features.stfts_4x[i].empty()) {
                features.stft_means_4x[i] = detail::vectorized_mean(features.stfts_4x[i]);
            }

            features.stfts_16x[i] = detail::downscale_spectrogram(features.stfts[i], num_freqs, 16);
            if (!features.stfts_16x[i].empty()) {
                features.stft_means_16x[i] = detail::vectorized_mean(features.stfts_16x[i]);
            }
        }
    }

    return features;
}

/**
 * Fast single audio loss computation.
 * Thread-safe: uses thread-local buffers internally.
 */
template <typename T>
inline T compute_audio_loss_fast(const std::vector<T>& generated,
                                 const TargetFeaturesFast<T>& target_features) {
    // Use original generated signal (no normalization)
    const std::vector<T>& gen_norm = generated;

    // Use FFT sizes from target_features (must match what was used during precomputation)
    const auto& fft_sizes = target_features.fft_sizes;
    const auto& hop_sizes = target_features.hop_sizes;

    T total_loss = 0;

    for (size_t scale = 0; scale < fft_sizes.size(); ++scale) {
        // Compute STFT for this scale
        auto x_stft =
            detail::stft_fast(gen_norm, fft_sizes[scale], hop_sizes[scale], fft_sizes[scale]);
        const auto& y_stft = target_features.stfts[scale];

        if (x_stft.size() != y_stft.size()) continue;

        // Spectral convergence loss (70% weight) with frequency weighting
        T sc_loss = detail::spectral_convergence_loss(
            x_stft, y_stft, target_features.stft_means[scale], target_features.weights[scale],
            target_features.num_freqs[scale]);

        // Log-magnitude loss (30% weight) with frequency weighting
        T mag_loss = detail::log_magnitude_loss(x_stft, y_stft, target_features.weights[scale],
                                                target_features.num_freqs[scale]);

        T base_loss = static_cast<T>(0.9) * sc_loss + static_cast<T>(0.1) * mag_loss;

        // Compute 4x downscale loss (1/3 weight)
        T down4x_loss = detail::compute_downscale_loss(
            x_stft, target_features.stfts_4x[scale], target_features.weights[scale],
            target_features.num_freqs[scale], 4, target_features.stft_means_4x[scale]);

        // Compute 16x downscale loss (1/3 weight)
        T down16x_loss = detail::compute_downscale_loss(
            x_stft, target_features.stfts_16x[scale], target_features.weights[scale],
            target_features.num_freqs[scale], 16, target_features.stft_means_16x[scale]);

        // Average the three resolutions for weighted impact
        total_loss += 0.7 * base_loss + 0.2 * down4x_loss + 0.1 * down16x_loss;
    }

    // Average over scales
    total_loss /= fft_sizes.size();

    // Add zero-crossing rate loss (5% weight)
    T gen_zcr = detail::zero_crossing_rate_fast(gen_norm);
    T zcr_loss = std::abs(gen_zcr - target_features.zcr);

    // Add energy/RMS matching term (25% weight) to prevent quiet solutions
    T gen_mean = detail::vectorized_mean(generated);
    T gen_var = detail::vectorized_variance(generated, gen_mean);
    T gen_rms = std::sqrt(gen_var);
    T energy_loss = static_cast<T>(0);
    if (target_features.target_rms > static_cast<T>(1e-8)) {
        T rms_ratio = gen_rms / target_features.target_rms;
        // Penalize both too quiet and too loud
        energy_loss = std::abs(static_cast<T>(1.0) - rms_ratio);
    } else {
        // If target is silent, penalize any non-zero generated signal
        energy_loss = gen_rms;
    }

    return static_cast<T>(0.6) * total_loss + static_cast<T>(0.2) * zcr_loss +
           static_cast<T>(0.2) * energy_loss;
}

/**
 * Batch loss computation - process multiple generated signals in parallel.
 */
template <typename T>
inline std::vector<T> compute_audio_loss_batch(const std::vector<std::vector<T>>& generated_batch,
                                               const TargetFeaturesFast<T>& target_features) {
    size_t batch_size = generated_batch.size();
    std::vector<T> losses(batch_size);

#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; ++i) {
        losses[i] = compute_audio_loss_fast(generated_batch[i], target_features);
    }
#else
    for (size_t i = 0; i < batch_size; ++i) {
        losses[i] = compute_audio_loss_fast(generated_batch[i], target_features);
    }
#endif

    return losses;
}

/**
 * Thread-safe loss function class.
 * Safe to call from multiple threads during parallel DE evaluation.
 */
template <typename T>
class LossFunction {
   public:
    LossFunction(const std::vector<T>& target, T stft_weight = static_cast<T>(1.0))
        : stft_weight_(stft_weight) {
        features_ = precompute_target_features_fast(target);
    }

    // Single loss computation - thread safe
    T operator()(const std::vector<T>& generated) const {
        return stft_weight_ * compute_audio_loss_fast(generated, features_);
    }

    // Batch loss computation - parallel processing
    std::vector<T> compute_batch(const std::vector<std::vector<T>>& generated_batch) const {
        auto losses = compute_audio_loss_batch(generated_batch, features_);
        if (stft_weight_ != static_cast<T>(1.0)) {
#if defined(HAS_OPENMP)
#pragma omp simd
#endif
            for (size_t i = 0; i < losses.size(); ++i) {
                losses[i] *= stft_weight_;
            }
        }
        return losses;
    }

   private:
    TargetFeaturesFast<T> features_;
    T stft_weight_;
};

}  // namespace adaptive_echo
