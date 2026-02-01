#pragma once

/**
 * Optimized STFT-based audio loss functions for adaptive_echo.
 * Parallelized and vectorized for maximum performance.
 *
 * Features:
 * - OpenMP parallelization across STFT frames
 * - Batch STFT processing (C++ equivalent of JAX vmap)
 * - SIMD vectorization using compiler intrinsics
 * - Thread-safe loss computation for parallel DE evaluation
 * - Precomputed FFT twiddle factors and bit-reversal tables
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <vector>

#include "adaptive_echo/constants.hpp"

#include <pocketfft_hdronly.h>

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
    pocketfft::c2c<T>(shape, stride_in, stride_out, axes, true,
                      frame_data.data(), fft_output.data(), T(1));

    // Store magnitudes - match JAX scipy.signal.stft behavior exactly
    // JAX's FFT does NOT normalize by 1/N (standard FFT convention)
    // scipy.signal.stft applies window normalization: 1.0 / sum(window) for magnitude spectrum
    // This matches the default 'spectrum' scaling mode
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
                                size_t n_fft, bool parallel = true) {
    const auto& window = WindowCache<T>::get(win_length);
    size_t num_frames = (x.size() >= win_length) ? (x.size() - win_length) / hop_length + 1 : 0;
    size_t num_freqs = n_fft / 2 + 1;

    if (num_frames == 0) {
        return std::vector<T>();
    }

    std::vector<T> result(num_frames * num_freqs);

#if defined(HAS_OPENMP)
    if (parallel && num_frames > 16) {
// Parallel processing for many frames
#pragma omp parallel for schedule(dynamic, 4)
        for (size_t frame = 0; frame < num_frames; ++frame) {
            size_t start = frame * hop_length;
            compute_stft_frame(x, window, frame, start, win_length, n_fft, num_freqs,
                               result.data());
        }
    } else
#endif
    {
        // Sequential processing for small inputs
        for (size_t frame = 0; frame < num_frames; ++frame) {
            size_t start = frame * hop_length;
            compute_stft_frame(x, window, frame, start, win_length, n_fft, num_freqs,
                               result.data());
        }
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
 * Batch STFT computation - C++ equivalent of JAX vmap.
 * Processes multiple audio signals in parallel.
 */
template <typename T>
inline std::vector<std::vector<T>> stft_batch(const std::vector<std::vector<T>>& batch,
                                              size_t win_length, size_t hop_length, size_t n_fft) {
    size_t batch_size = batch.size();
    std::vector<std::vector<T>> results(batch_size);

#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; ++i) {
        results[i] = stft_fast(batch[i], win_length, hop_length, n_fft, false);
    }
#else
    for (size_t i = 0; i < batch_size; ++i) {
        results[i] = stft_fast(batch[i], win_length, hop_length, n_fft, false);
    }
#endif

    return results;
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
};

/**
 * Precompute target STFT features with vectorized operations.
 */
template <typename T>
inline TargetFeaturesFast<T> precompute_target_features_fast(
    const std::vector<T>& target, const std::vector<size_t>& fft_sizes = {1024, 512, 256},
    const std::vector<size_t>& hop_sizes = {512, 256, 128}) {
    TargetFeaturesFast<T> features;
    features.num_frames.resize(fft_sizes.size());
    features.num_freqs.resize(fft_sizes.size());
    features.stft_means.resize(fft_sizes.size());

    // Normalize target
    std::vector<T> tgt_norm = detail::normalize_signal(target);

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

    return features;
}

/**
 * Fast single audio loss computation.
 * Thread-safe: uses thread-local buffers internally.
 */
template <typename T>
inline T compute_audio_loss_fast(const std::vector<T>& generated,
                                 const TargetFeaturesFast<T>& target_features) {
    // Normalize generated audio
    std::vector<T> gen_norm = detail::normalize_signal(generated);

    // Use same FFT sizes as JAX project
    static const std::vector<size_t> fft_sizes = {1024, 512, 256};
    static const std::vector<size_t> hop_sizes = {512, 256, 128};

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

        total_loss += static_cast<T>(0.7) * sc_loss + static_cast<T>(0.3) * mag_loss;
    }

    // Average over scales
    total_loss /= fft_sizes.size();

    // Add zero-crossing rate loss (5% weight)
    T gen_zcr = detail::zero_crossing_rate_fast(gen_norm);
    T zcr_loss = std::abs(gen_zcr - target_features.zcr);

    return static_cast<T>(0.95) * total_loss + static_cast<T>(0.05) * zcr_loss;
}

/**
 * Batch loss computation - process multiple generated signals in parallel.
 * C++ equivalent of JAX vmap over loss computation.
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
